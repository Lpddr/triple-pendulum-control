"""Reward-driven cross-entropy policy search on the distilled actor.

Each candidate runs closed-loop in MuJoCo from randomized initial states.
No teacher actions are used in this stage: fitness is the task reward measured
from the candidate's own rollouts. Deployment still uses only the saved actor.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
from mujoco import rollout
import numpy as np
import torch
from algorithms.gps.trajectory import Plant


def score(plant,weights,reference,initial,steps=400):
    candidates=len(weights); episodes=len(initial)
    x=np.tile(initial,(candidates,1))
    cur=np.zeros(len(x)); longest=cur.copy(); shaping=cur.copy(); cartmax=cur.copy()
    penalty=cur.copy()
    for t in range(steps):
        phase=min(t,len(reference)-1)
        params=np.repeat(weights[:,phase,:],episodes,axis=0)
        u=np.clip(np.sum(params[:,:8]*(x-reference[phase]),axis=1)+params[:,8],-1,1)
        states=np.c_[np.zeros(len(x)),x]
        controls=np.broadcast_to(u[:,None,None],(len(x),plant.frame_skip,1)).copy()
        result,_=rollout.rollout(plant.model,plant.pool,states,controls,persistent_pool=True)
        x=result[:,-1,1:]
        a=np.arctan2(np.sin(np.cumsum(x[:,1:4],axis=1)),np.cos(np.cumsum(x[:,1:4],axis=1)))
        v=np.cumsum(x[:,5:],axis=1)
        good=(np.max(abs(a),axis=1)<np.deg2rad(15))&(np.max(abs(v),axis=1)<1)&(abs(x[:,0])<.95)
        cur=(cur+1)*good; longest=np.maximum(cur,longest)
        # Dense upright reward aids search before an episode passes 10 seconds.
        shaping+=np.exp(-2*np.sum(a*a,axis=1)-.05*np.sum(v*v,axis=1))
        penalty+=np.maximum(abs(x[:,0])-.9,0)*50
        cartmax=np.maximum(cartmax,abs(x[:,0]))
    reward=longest*plant.dt+shaping/steps-penalty/steps
    return reward.reshape(candidates,episodes).mean(axis=1),longest.reshape(candidates,episodes)*plant.dt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='runs/gps/phase/policy.pt')
    ap.add_argument('--out',default='runs/gps/refined')
    ap.add_argument('--generations',type=int,default=30)
    ap.add_argument('--population',type=int,default=32)
    ap.add_argument('--episodes',type=int,default=12)
    ap.add_argument('--width',type=float,default=.05)
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args()
    torch.set_num_threads(1)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    base=checkpoint['policy']['embedding.weight'].numpy().copy()
    reference=checkpoint['policy']['reference'].numpy()
    rng=np.random.default_rng(args.seed)
    initial=np.tile(np.array([0,np.pi,0,0,0,0,0,0.]),(args.episodes,1))
    initial[:,:4]+=rng.uniform(-args.width,args.width,(args.episodes,4))
    initial[:,4:]+=rng.normal(0,args.width,(args.episodes,4)); initial[0]=[0,np.pi,0,0,0,0,0,0]
    # Six smoothly varying phase blocks, eight learned feedback scales per block.
    blocks=6
    locations=np.linspace(0,1,blocks)
    phase=np.linspace(0,1,len(base))
    blend=np.maximum(1-abs(phase[:,None]-locations[None,:])*(blocks-1),0)
    mean=np.zeros((blocks,8)); std=np.full_like(mean,.12)
    best_params=mean.copy(); best=-np.inf
    plant=Plant(threads=6)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    start=time.time()
    with (out/'learning.jsonl').open('w',encoding='utf-8') as log:
        for gen in range(args.generations):
            params=rng.normal(mean,std,(args.population,blocks,8))
            params[0]=best_params; params[1]=mean
            scales=np.exp(np.clip(np.einsum('tb,pbd->ptd',blend,params),-.6,.6))
            weights=np.repeat(base[None],args.population,axis=0)
            weights[:,:,:8]*=scales
            rewards,holds=score(plant,weights,reference,initial)
            elite=np.argsort(rewards)[-max(4,args.population//4):]
            winner=int(np.argmax(rewards))
            if rewards[winner]>best:
                best=float(rewards[winner]);best_params=params[winner].copy()
                checkpoint['policy']['embedding.weight']=torch.from_numpy(weights[winner].copy())
                checkpoint['refinement']='cross_entropy_reward_policy_search'
                checkpoint['training_seed']=args.seed
                checkpoint['training_reward']=best
                torch.save(checkpoint,out/'policy.pt')
            mean=.3*mean+.7*params[elite].mean(axis=0)
            std=np.maximum(.3*std+.7*params[elite].std(axis=0),.025)
            row=dict(generation=gen,reward_max=float(rewards.max()),reward_mean=float(rewards.mean()),
                     training_passing=int((holds[winner]>=10).sum()),episodes=args.episodes,
                     hold_min=float(holds[winner].min()),seconds=round(time.time()-start,1))
            print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()


if __name__=='__main__':main()
