"""Auditable evaluation: one network, one reset per episode, cart actuation only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from algorithms.gps.policy import Policy, PhasePolicy
ROOT=Path(__file__).resolve().parents[1]


def is_upright(qpos,qvel,angle_deg=15.,angular_speed=1.):
    """All absolute link angles, not relative hinge angles or tip height alone."""
    angles=np.arctan2(np.sin(np.cumsum(qpos[1:])),np.cos(np.cumsum(qpos[1:])))
    speeds=np.cumsum(qvel[1:])
    return bool(np.max(abs(angles))<=np.deg2rad(angle_deg)
                and np.max(abs(speeds))<=angular_speed and abs(qpos[0])<=.95)


def evaluate(args):
    if args.episodes<1 or args.seconds<=0 or args.width<0 or args.velocity<0:
        raise ValueError('episodes/seconds must be positive; initial noise must be nonnegative')
    torch.set_num_threads(1)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    phase=ckpt.get('architecture')=='phase_affine'
    policy=PhasePolicy(ckpt['phases']) if phase else Policy(ckpt['hidden'])
    policy.load_state_dict(ckpt['policy']); policy.eval()
    xml=ROOT/'triple_pendulum/assets/inverted_triple_pendulum.xml'
    model=mujoco.MjModel.from_xml_path(str(xml))
    data=mujoco.MjData(model)
    frame_skip=int(ckpt['frame_skip']); dt=model.opt.timestep*frame_skip
    assert model.nu==1 and model.actuator_trnid[0,0]==model.joint('slider').id
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    renderer=writer=None
    if args.video:
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(model,height=720,width=960)
        writer=imageio.get_writer(str(out/'demo.mp4'),fps=round(1/dt),codec='libx264',quality=8)
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[0,0,0]; camera.distance=6.8; camera.azimuth=90; camera.elevation=-7
    rows=[]
    try:
        for ep in range(args.episodes):
            rng=np.random.default_rng(args.seed+ep)
            mujoco.mj_resetData(model,data)
            # These are the ONLY qpos/qvel writes in an evaluation episode.
            data.qpos[:]=[0,np.pi,0,0]
            if args.start=='large': data.qpos[1:]=rng.uniform(-np.pi,np.pi,3)
            else: data.qpos[:]+=rng.uniform(-args.width,args.width,4)
            data.qvel[:]=rng.normal(0,args.velocity,4)
            mujoco.mj_forward(model,data)
            initial=np.r_[data.qpos.copy(),data.qvel.copy()]
            cur=longest=post_cur=post_longest=pre_longest=0
            max_cart=0.; max_force=0.; first_hold=None; resets=1
            history=[]
            for t in range(round(args.seconds/dt)):
                state=np.r_[data.qpos,data.qvel]
                with torch.inference_mode():
                    obs=torch.tensor(state,dtype=torch.float64 if phase else torch.float32)[None]
                    action=float((policy.action(obs,t) if phase else policy.action(obs)).item())
                data.ctrl[0]=action
                force=args.force if args.push_at<=t*dt<args.push_at+args.push_duration else 0.
                # Physical external force on top link; never change state to kick it.
                data.xfrc_applied[:]=0
                data.xfrc_applied[model.body('pole3').id,0]=force
                # Score every physics tick, not only sampled control frames.
                for _ in range(frame_skip):
                    mujoco.mj_step(model,data)
                    good=is_upright(data.qpos,data.qvel,args.angle_deg,args.angular_speed)
                    cur=cur+1 if good else 0; longest=max(longest,cur)
                    if data.time<args.push_at:
                        pre_longest=max(pre_longest,cur)
                    if data.time>=args.push_at+args.push_duration:
                        post_cur=post_cur+1 if good else 0; post_longest=max(post_longest,post_cur)
                    if first_hold is None and cur*model.opt.timestep>=10-1e-8:
                        first_hold=float(data.time)
                    max_cart=max(max_cart,abs(float(data.qpos[0])))
                mujoco.mj_forward(model,data)
                max_force=max(max_force,abs(action)*model.actuator_gear[0,0])
                history.append(np.r_[data.time,data.qpos.copy(),data.qvel.copy(),action,force,cur*model.opt.timestep])
                if writer is not None and ep==0:
                    renderer.update_scene(data,camera=camera)
                    frame=renderer.render()
                    from PIL import Image,ImageDraw,ImageFont
                    im=Image.fromarray(frame); draw=ImageDraw.Draw(im)
                    font=ImageFont.load_default(size=18)
                    draw.rectangle((0,0,960,95),fill=(12,16,25))
                    draw.text((18,10),f'TRIPLE PENDULUM | single learned policy | t={data.time:05.2f} s',fill='white',font=font)
                    draw.text((18,38),f'Upright hold: {cur*model.opt.timestep:05.2f} s | cart: {action*500:+07.1f} N | push: {force:+.1f} N',fill='white',font=font)
                    draw.text((18,66),'One continuous episode / cart actuation only / no state correction',fill='white',font=font)
                    writer.append_data(np.asarray(im))
            hold=longest*model.opt.timestep; post=post_longest*model.opt.timestep
            row=dict(seed=args.seed+ep,start=args.start,initial_state=initial.tolist(),hold_seconds=hold,
                     pre_push_hold_seconds=pre_longest*model.opt.timestep,
                     post_push_hold_seconds=post,passing=hold>=10-1e-8,
                     recovery_passing=post>=10-1e-8 if args.force else None,
                     first_10s_hold_completed_at=first_hold,max_cart_m=max_cart,max_force_N=max_force,resets=resets)
            rows.append(row)
            np.savez_compressed(out/f'episode_{ep:03}.npz',trajectory=np.asarray(history),initial_state=initial,
                                columns=np.array(['time','cart_x','q1','q2','q3','cart_v','dq1','dq2','dq3','action','push_N','hold_s']))
            print(json.dumps(row),flush=True)
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),config=vars(args),dt=dt,
                passing=sum(r['passing'] for r in rows),episodes=len(rows),
                recovery_passing=sum(bool(r['recovery_passing']) for r in rows),rows=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS',report['passing'],'/',len(rows),'RECOVERY',report['recovery_passing'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',default='runs/gps/evaluation')
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--start',choices=['hanging','large'],default='hanging')
    p.add_argument('--width',type=float,default=.03)
    p.add_argument('--velocity',type=float,default=.03)
    p.add_argument('--seconds',type=float,default=35)
    p.add_argument('--force',type=float,default=0.)
    p.add_argument('--push-at',type=float,default=18.)
    p.add_argument('--push-duration',type=float,default=.2)
    p.add_argument('--angle-deg',type=float,default=15.)
    p.add_argument('--angular-speed',type=float,default=1.)
    p.add_argument('--video',action='store_true')
    evaluate(p.parse_args())
