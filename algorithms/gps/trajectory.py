"""MuJoCo trajectory-centric policy improvement, used only during training.

Finite differences use the actual RK4 / 20 Hz plant. No approximate equations,
extra actuators, state interventions within rollouts, or controller at evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
from mujoco import rollout
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from algorithms.sac.sac_model import Actor


class Plant:
    def __init__(self, frame_skip=5, threads=4):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'triple_pendulum/assets/inverted_triple_pendulum.xml'))
        self.data = mujoco.MjData(self.model)
        self.pool = [mujoco.MjData(self.model) for _ in range(threads)]
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip

    def step(self, x, u):
        self.data.qpos[:] = x[:4]
        self.data.qvel[:] = x[4:]
        self.data.qacc_warmstart[:] = 0
        self.data.ctrl[:] = u
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        return np.r_[self.data.qpos, self.data.qvel]

    def derivatives(self, x, u, eps=1e-5):
        n = len(u)
        z = np.c_[x[:-1], u]
        perturb = np.r_[np.eye(9), -np.eye(9)] * eps
        z = (z[:, None, :] + perturb[None]).reshape(-1, 9)
        states = np.c_[np.zeros(len(z)), z[:, :8]]
        controls = np.repeat(z[:, None, 8:], self.frame_skip, axis=1)
        result, _ = rollout.rollout(self.model, self.pool, states, controls, persistent_pool=True)
        result = result[:, -1, 1:].reshape(n, 18, 8)
        jac = ((result[:, :9] - result[:, 9:]) / (2 * eps)).transpose(0, 2, 1)
        return jac[:, :, :8], jac[:, :, 8:]

    def run(self, x0, u, reference=None, gain=None, delta=None, alpha=1):
        x = np.empty((len(u)+1, 8))
        actual = np.empty_like(u)
        x[0] = x0
        for t in range(len(u)):
            a = u[t].copy()
            if gain is not None:
                a += gain[t] @ (x[t]-reference[t]) + alpha * delta[t]
            actual[t] = np.clip(a, -1, 1)
            x[t+1] = self.step(x[t], actual[t])
        return x, actual


def terminal_matrix(plant):
    x = np.zeros((2, 8))
    a, b = plant.derivatives(x, np.zeros((1, 1)))
    a, b = a[0], b[0]
    q = np.diag([2., 8., 8., 8., .2, .2, .2, .2])
    p = q.copy()
    for _ in range(3000):
        k = -np.linalg.solve(.1 * np.eye(1) + b.T @ p @ b, b.T @ p @ a)
        nxt = q + a.T @ p @ (a + b @ k)
        nxt = (nxt + nxt.T) / 2
        if np.max(abs(nxt-p)) < 1e-8:
            break
        p = nxt
    return p, k


class Cost:
    def __init__(self, goal, terminal, n):
        self.goal = goal
        self.q = np.diag([.15, .02, .02, .02, .003, .003, .003, .003])
        self.terminal = terminal
        self.r = .02
        self.n = n

    def evaluate(self, x, u, deriv=False):
        error = x - self.goal
        q = np.repeat(self.q[None], len(x), axis=0)
        q[-1] = self.terminal
        lx = np.einsum('tij,tj->ti', q, error)
        costs = .5*np.einsum('ti,ti->t', error, lx)
        over = np.maximum(abs(x[:, 0]) - .90, 0)
        costs += 5000 * over**2
        lx[:, 0] += 10000 * over * np.sign(x[:, 0])
        q[:, 0, 0] += 10000 * (over > 0)
        total = costs.sum() + .5 * self.r * (u**2).sum()
        if deriv:
            return lx, q, self.r*u, self.r
        return float(total)


def backward(a, b, lx, lxx, lu, luu, u, reg):
    n = len(a)
    gains, delta = np.empty((n, 1, 8)), np.empty((n, 1))
    vx, vxx = lx[-1].copy(), lxx[-1].copy()
    for t in reversed(range(n)):
        qx = lx[t] + a[t].T @ vx
        qu = lu[t] + b[t].T @ vx
        qxx = lxx[t] + a[t].T @ vxx @ a[t]
        quu = float((b[t].T @ vxx @ b[t]).item()) + luu
        qux = b[t].T @ vxx @ a[t]
        denom = quu + reg
        if denom <= 0 or not np.isfinite(denom):
            return None, None
        d = -qu / denom
        # Active-set solution of the scalar box-constrained local Q problem.
        delta[t] = np.clip(d, -1-u[t], 1-u[t])
        gains[t] = -qux / denom if np.all(abs(delta[t]-d)<1e-10) else 0
        k = gains[t]
        d = delta[t]
        vx = qx + (k.T @ qu).ravel() + (qux.T @ d).ravel() + (k.T * quu @ d).ravel()
        vxx = qxx + k.T * quu @ k + k.T @ qux + qux.T @ k
        vxx = (vxx+vxx.T)/2
    return gains, delta


def improve(plant, x0, u, cost, iterations=150, verbose=True):
    x, u = plant.run(x0, u)
    value = cost.evaluate(x, u)
    reg = 1.
    start = time.time()
    for iteration in range(iterations):
        a, b = plant.derivatives(x, u)
        lx, lxx, lu, luu = cost.evaluate(x, u, True)
        accepted = False
        for _ in range(12):
            k, d = backward(a, b, lx, lxx, lu, luu, u, reg)
            if k is not None:
                for alpha in [1., .5, .25, .1, .05, .01]:
                    xn, un = plant.run(x0, u, x, k, d, alpha)
                    vn = cost.evaluate(xn, un)
                    if np.isfinite(vn) and vn < value:
                        x, u, value = xn, un, vn
                        accepted = True
                        break
            if accepted:
                reg = max(reg/2, 1e-6)
                break
            reg *= 10
        if verbose and iteration % 10 == 0:
            print(f'iter={iteration} cost={value:.4f} terminal={np.linalg.norm(x[-1]-cost.goal):.4f} cart={abs(x[:,0]).max():.3f} reg={reg:.2g} seconds={time.time()-start:.1f}', flush=True)
        if reg > 1e12:
            break
    a, b = plant.derivatives(x, u)
    k, _ = backward(a,b,*cost.evaluate(x,u,True),u,1e-5)
    return x, u, k, value


def seed_from_sac(plant, checkpoint, steps, seed):
    actor = Actor(12,1)
    actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    actor.eval()
    x = np.array([0,np.pi,0,0,0,0,0,0.],float)
    rng = np.random.default_rng(seed)
    if seed:
        x[:4] += rng.uniform(-.1,.1,4)
    xs, us = [x.copy()], []
    for t in range(steps):
        mujoco.mj_forward(plant.model,plant.data)
        obs = np.r_[x[0],np.sin(x[1:4]),np.cos(x[1:4]),np.clip(x[4:],-10,10),np.clip(plant.data.qfrc_constraint[0],-10,10)]
        with torch.inference_mode():
            u=actor.deterministic(torch.tensor(obs,dtype=torch.float32)[None]).numpy()[0]
        x=plant.step(x,u)
        xs.append(x.copy()); us.append(u)
    return np.array(xs),np.array(us)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--steps',type=int,default=120)
    ap.add_argument('--iterations',type=int,default=150)
    ap.add_argument('--out',default='runs/gps/trajectory.npz')
    ap.add_argument('--checkpoint',default='runs/sac_swingup_mixed/sac_swingup_final.pt')
    args=ap.parse_args()
    torch.set_num_threads(1)
    plant=Plant()
    p,terminal_k=terminal_matrix(plant)
    x,u=seed_from_sac(plant,args.checkpoint,args.steps,args.seed)
    heights=.6*np.cos(np.cumsum(x[:,1:4],axis=1)).sum(axis=1)
    candidates=np.where(heights[25:]>1.65)[0]+25
    end=int(candidates[np.argmin(np.linalg.norm(x[candidates,4:],axis=1))]) if len(candidates) else int(np.argmax(heights[25:])+25)
    print('seed endpoint',end,'height',heights[end],'state',x[end],flush=True)
    u=u[:end]
    goal=np.zeros(8); goal[1:4]=np.round(x[end,1:4]/(2*np.pi))*2*np.pi
    cost=Cost(goal,p,len(u))
    x,u,k,value=improve(plant,x[0],u,cost,args.iterations)
    out=Path(args.out); out.parent.mkdir(exist_ok=True,parents=True)
    np.savez(out,x=x,u=u,k=k,goal=goal,terminal_k=terminal_k,terminal_p=p,cost=value,dt=plant.dt)
    print('saved',out,'terminal',x[-1]-goal,flush=True)


if __name__=='__main__': main()
