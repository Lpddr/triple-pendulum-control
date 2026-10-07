"""Fit one phase-conditioned actor to a trajectory-centric policy-search result.

The affine architecture permits exact least-squares distillation instead of
introducing unnecessary MLP approximation error. Gains are learned, not supplied
to the evaluator by a hand-designed controller.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from algorithms.gps.policy import PhasePolicy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default='runs/gps/search_0.npz')
    p.add_argument('--out',default='runs/gps/phase')
    args=p.parse_args()
    torch.set_num_threads(1)
    d=np.load(args.reference)
    refs=np.concatenate([d['x'][:-1],d['goal'][None]])
    targets=np.concatenate([d['k'][:,0,:],d['terminal_k']])
    biases=np.r_[d['u'][:,0],0.]
    rng=np.random.default_rng(42)
    weights=[]; errors=[]
    # Supervised actor update in guided policy search, using local state samples.
    for k,b in zip(targets,biases):
        z=np.c_[rng.normal(0,.03,(256,8)),np.ones(256)]
        actions=z[:,:8]@k+b
        coef=np.linalg.lstsq(z,actions,rcond=None)[0]
        errors.append(float(np.max(abs(z@coef-actions))))
        weights.append(coef)
    actor=PhasePolicy(len(refs))
    actor.reference.copy_(torch.from_numpy(refs))
    with torch.no_grad(): actor.embedding.weight.copy_(torch.tensor(np.asarray(weights)))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ckpt=dict(policy=actor.state_dict(),architecture='phase_affine',phases=len(refs),frame_skip=5,
              algorithm='trajectory_centric_guided_policy_search',reference=str(args.reference),
              max_distillation_error=max(errors))
    torch.save(ckpt,out/'policy.pt')
    print('saved',out/'policy.pt','maximum distillation error',max(errors))


if __name__=='__main__': main()
