"""Model-based guided policy search: reward optimization, sampling and distillation.

Trajectory teachers are training-only. The exported stationary neural network
maps the observed positions and velocities directly to one cart force.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from algorithms.gps.trajectory import Plant, Cost, improve, terminal_matrix, backward
from algorithms.gps.policy import Policy
import numpy as np
import torch


def teachers(plant,reference,count,seed,out):
    rng=np.random.default_rng(seed)
    d=np.load(reference)
    result=[dict(d)]
    for i in range(count):
        x0=d['x'][0].copy()
        x0[:4]+=rng.uniform(-.15,.15,4)
        x0[4:]+=rng.normal(0,.15,4)
        c=Cost(d['goal'],d['terminal_p']*10,len(d['u']))
        c.q=np.diag([1.,.01,.01,.01,.001,.001,.001,.001])
        x,u,k,v=improve(plant,x0,d['u'].copy(),c,70,False)
        final=x[-1].copy()
        # Validation in the actual nonlinear plant, with no state reset.
        for t in range(200):
            final=plant.step(final,np.clip(d['terminal_k']@(final-d['goal']),-1,1))
        success=np.linalg.norm(final-d['goal'])<.1 and np.max(abs(x[:,0]))<.95
        print('teacher',i,'success',success,'endpoint',np.linalg.norm(x[-1]-d['goal']),flush=True)
        if success:
            item=dict(x=x,u=u,k=k,goal=d['goal'],terminal_k=d['terminal_k'],terminal_p=d['terminal_p'])
            result.append(item)
            np.savez(out/f'teacher_{i:03}.npz',**item)
    return result


def dataset(items,seed):
    rng=np.random.default_rng(seed)
    states=[]; actions=[]
    for d in items:
        n=len(d['u'])
        # Local samples around trajectories; mirror symmetry doubles coverage.
        for _ in range(100):
            noise=rng.normal(size=(n,8))*np.array([.008,.008,.008,.008,.025,.025,.025,.025])
            x=d['x'][:-1]+noise
            u=d['u']+np.einsum('tij,tj->ti',d['k'],noise)
            states.append(x); actions.append(u)
    # The stationary goal region is part of the same supervised objective.
    n=max(60000,sum(len(x) for x in states)//2)
    noise=rng.normal(size=(n,8))*rng.choice([.02,.06,.12],size=(n,1))*np.array([2,1,1,1,3,3,3,3])
    u=noise@items[0]['terminal_k'].T
    states.append(noise); actions.append(u)
    return torch.tensor(np.concatenate(states),dtype=torch.float32),torch.tensor(np.clip(np.concatenate(actions),-1,1),dtype=torch.float32)


def evaluate(policy,plant,episodes=20,width=.03,seed=1000):
    rng=np.random.default_rng(seed); holds=[]; maxc=[]
    for ep in range(episodes):
        x=np.array([0,np.pi,0,0,0,0,0,0.])
        x[:4]+=rng.uniform(-width,width,4); x[4:]+=rng.normal(0,width,4)
        longest=cur=0; cart=0
        for t in range(400):
            with torch.inference_mode():
                u=policy.action(torch.tensor(x,dtype=torch.float32)[None]).numpy()[0]
            x=plant.step(x,u)
            a=np.arctan2(np.sin(np.cumsum(x[1:4])),np.cos(np.cumsum(x[1:4])))
            good=np.max(abs(a))<np.deg2rad(15) and np.max(abs(np.cumsum(x[5:])))<1.0 and abs(x[0])<.95
            cur=cur+1 if good else 0; longest=max(longest,cur); cart=max(cart,abs(x[0]))
        holds.append(longest*plant.dt); maxc.append(cart)
    return dict(passing=sum(h>=10 for h in holds),episodes=episodes,hold_mean=float(np.mean(holds)),hold_max=float(np.max(holds)),cart_max=float(np.max(maxc)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',default='runs/gps/search_0.npz')
    ap.add_argument('--out',default='runs/gps/student')
    ap.add_argument('--teachers',type=int,default=24)
    ap.add_argument('--updates',type=int,default=16000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--reuse',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    plant=Plant()
    if args.reuse:
        items=[dict(np.load(args.reference))]+[dict(np.load(f)) for f in sorted(out.glob('teacher_*.npz'))]
    else: items=teachers(plant,args.reference,args.teachers,args.seed,out)
    x,y=dataset(items,args.seed)
    print('samples',len(x),'teachers',len(items),flush=True)
    policy=Policy()
    opt=torch.optim.Adam(policy.parameters(),lr=1e-3)
    best=-1; start=time.time()
    for step in range(args.updates):
        ids=torch.randint(len(x),(1024,))
        pred=policy(x[ids])
        # Clipped teacher targets, unclipped predictions keep gradients alive.
        loss=((pred-y[ids])**2).mean()
        local=torch.randn((128,8))*.001
        local_target=local@torch.tensor(items[0]['terminal_k'].T,dtype=torch.float32)
        loss=loss+100*((policy(local)-local_target)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1)%2000==0:
            metrics=evaluate(policy,plant,episodes=10,width=.03)
            print('update',step+1,'loss',float(loss.detach()),'seconds',round(time.time()-start,1),metrics,flush=True)
            score=metrics['passing']*100+metrics['hold_mean']
            ckpt=dict(policy=policy.state_dict(),algorithm='guided_policy_search',hidden=128,step=step+1,metrics=metrics,frame_skip=5)
            torch.save(ckpt,out/'latest.pt')
            if score>best:
                torch.save(ckpt,out/'best.pt'); best=score
            for group in opt.param_groups: group['lr']*=.8
    torch.save(ckpt,out/'final.pt')


if __name__=='__main__': main()
