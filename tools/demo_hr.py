"""Real-time demonstration. One episode, one policy, no mid-episode reset."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
import numpy as np
import torch
from algorithms.gps.policy import PhasePolicy
from tools.evaluate_gps import is_upright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='checkpoints/hr/policy.pt')
    p.add_argument('--seed',type=int,default=2000)
    p.add_argument('--width',type=float,default=.05)
    p.add_argument('--seconds',type=float,default=35)
    p.add_argument('--force',type=float,default=5)
    args=p.parse_args()
    torch.set_num_threads(1)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    actor=PhasePolicy(checkpoint['phases']);actor.load_state_dict(checkpoint['policy']);actor.eval()
    root=Path(__file__).resolve().parents[1]
    m=mujoco.MjModel.from_xml_path(str(root/'triple_pendulum/assets/inverted_triple_pendulum.xml'))
    d=mujoco.MjData(m)
    rng=np.random.default_rng(args.seed)
    d.qpos[:]=np.array([0,np.pi,0,0])+rng.uniform(-args.width,args.width,4)
    d.qvel[:]=rng.normal(0,args.width,4)
    mujoco.mj_forward(m,d)
    dt=m.opt.timestep*checkpoint['frame_skip'];step=0;held=0
    with mujoco.viewer.launch_passive(m,d) as viewer:
        viewer.cam.lookat[:]=[0,0,0];viewer.cam.distance=6.8
        viewer.cam.azimuth=90;viewer.cam.elevation=-7
        while viewer.is_running() and step*dt<args.seconds:
            tick=time.perf_counter()
            x=torch.from_numpy(np.r_[d.qpos,d.qvel])[None]
            with torch.inference_mode():d.ctrl[0]=actor.action(x,step).item()
            d.xfrc_applied[:]=0
            force=args.force if 18<=step*dt<18.2 else 0
            d.xfrc_applied[m.body('pole3').id,0]=force
            for _ in range(checkpoint['frame_skip']):
                mujoco.mj_step(m,d)
                held=held+m.opt.timestep if is_upright(d.qpos,d.qvel) else 0
            mujoco.mj_forward(m,d);viewer.sync()
            if step%20==0:
                print(f't={d.time:5.2f}s  upright={held:5.2f}s  cart={d.qpos[0]:+.3f}m  push={force:+.1f}N',flush=True)
            step+=1
            time.sleep(max(0,dt-(time.perf_counter()-tick)))


if __name__=='__main__':main()
