"""SAC training for trip pend.
structure: 1. act in env, store transition -> 2. sample minibatch from
replay buffer -> 3. update critics, actor, alpha -> repeat.
no rollout phases, no gae, no clipping - a complete different loop than ppo

vectorized: n_envs step in parallel, but only ONE gradient update per
vector step (not per env). that was the big sps killer before - full
critic/actor/alpha update after every single mujoco step on cpu.
"""

import argparse
import os
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AutoresetMode
import wandb
import triple_pendulum # registers InvertedTriplePendulum-v0
try:
    from .sac_model import Actor, QCritic
except ImportError:
    from sac_model import Actor, QCritic

@dataclass
class cfg:
    env_id: str = "InvertedTriplePendulum-v0"
    reset_noise_scale: float = 0.05
    # swing-up task: curriculum reset + conditional termination + energy shaping
    swingup: bool = True
    reset_angle_limit: float = 1.2  # curriculum: 0.6 -> 1.2 -> pi
    near_prob: float = 0.0  # fraction of resets starting near upright
    near_width: float = 0.15
    capture_reward: bool = True  # region-gated reward: above capture_zone the
                                 # energy term is replaced by strong velocity
                                 # damping + height bonus (see env.py)
    capture_vel_w: float = 1.0  # v^2 weight inside the capture zone.  KEEP THIS
                                # SMALL (0.1): a heavy penalty on a state the
                                # agent cannot yet avoid teaches avoidance, not
                                # skill -- the value cliff at the zone entrance
                                # is what killed the first capture-reward run.  # half-width in rad of the near-upright branch
    init_from: str = ""  # warm-start the actor from a previous checkpoint
    frame_skip: int = 5  # physics steps per control step; 2 -> 50 Hz control
    joint_damping: float = 0.0  # >0: override hinge damping (cart keeps 0.05).
                                # Passive dissipation decelerates fly-through
                                # arrivals, widening the physically feasible
                                # capture basin. Legitimate plant modelling:
                                # real rigs have friction.
    max_episode_steps: int = 1000  # at frame_skip=2 use 2000+ (40 s episodes)
    near_vel_std: float = 0.2  # qvel noise of the near-upright branch; raise it
                               # to practice catching the top WITH momentum
    fixed_alpha: float = 0.0  # >0: pin alpha (no auto-tune). A warm-started
                              # actor is low-entropy, so the auto-tuner climbs
                              # alpha and the action jitter destroys capture.
    total_timesteps: int = 5_000_000
    n_envs: int = 16 # same as ppo - free env steps while the update still dominates
    buffer_size: int = 1_000_000
    batch_size: int = 256
    start_steps: int = 5_000 # pure random warmup to seed the buffer
    gamma: float = 0.99
    tau: float = 0.005 # soft target update rate
    lr: float = 3e-4
    hidden_dim: int = 256
    seed: int = 1
    save_dir: str = "runs/sac_swingup"  # never write into the repo root:
                                        # that is how the author's checkpoint
                                        # got clobbered last time
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = cfg()

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32) # terminations only
        self.ptr, self.full, self.size = 0, False, size

    def add_batch(self, o, a, r, no, d):
        # vector step dumps n_envs transitions at once
        n = o.shape[0]
        p = self.ptr
        if p + n <= self.size:
            sl = slice(p, p + n)
            self.obs[sl], self.act[sl], self.rew[sl] = o, a, r
            self.next_obs[sl], self.done[sl] = no, d
            self.ptr = p + n
            if self.ptr == self.size:
                self.full, self.ptr = True, 0
        else:
            # wrap around the end of the ring
            k = self.size - p
            self.add_batch(o[:k], a[:k], r[:k], no[:k], d[:k])
            self.add_batch(o[k:], a[k:], r[k:], no[k:], d[k:])

    def sample(self, batch_size, device):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)

def make_env():
    def thunk():
        env = gym.make(
            config.env_id,
            reset_noise_scale=config.reset_noise_scale,
            swingup=config.swingup,
            reset_angle_limit=config.reset_angle_limit,
            near_prob=config.near_prob,
            near_width=config.near_width,
            near_vel_std=config.near_vel_std,
            capture_reward=config.capture_reward,
            capture_vel_w=config.capture_vel_w,
            frame_skip=config.frame_skip,
            max_episode_steps=config.max_episode_steps,
        )
        if config.joint_damping > 0:
            env.unwrapped.model.dof_damping[1:] = config.joint_damping
        return env
    return thunk


def parse_args():
    p = argparse.ArgumentParser(description="SAC training (balance or swing-up)")
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--angle-limit", type=float, default=None,
                   help="curriculum angle bound in rad (0.6 / 1.2 / 3.14)")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--near-prob", type=float, default=None,
                   help="fraction of resets starting near upright (capture practice)")
    p.add_argument("--near-width", type=float, default=None,
                   help="half-width in rad of the near-upright reset branch")
    p.add_argument("--near-vel-std", type=float, default=None,
                   help="qvel std of the near-upright reset branch")
    p.add_argument("--capture-vel-w", type=float, default=None,
                   help="v^2 weight inside the capture zone (keep small, ~0.1)")
    p.add_argument("--no-capture-reward", action="store_true",
                   help="disable the region-gated capture reward")
    p.add_argument("--fixed-alpha", type=float, default=None,
                   help="pin the entropy temperature instead of auto-tuning")
    p.add_argument("--joint-damping", type=float, default=None,
                   help="override hinge damping (0 = keep XML value)")
    p.add_argument("--frame-skip", type=int, default=None,
                   help="physics steps per control step (2 = 50 Hz control)")
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start the actor from a previous checkpoint")
    p.add_argument("--no-swingup", action="store_true",
                   help="train the original balance-only task instead")
    a = p.parse_args()
    if a.total_timesteps is not None: config.total_timesteps = a.total_timesteps
    if a.angle_limit is not None:     config.reset_angle_limit = a.angle_limit
    if a.n_envs is not None:          config.n_envs = a.n_envs
    if a.seed is not None:            config.seed = a.seed
    if a.save_dir is not None:        config.save_dir = a.save_dir
    if a.near_prob is not None:       config.near_prob = a.near_prob
    if a.near_width is not None:      config.near_width = a.near_width
    if a.near_vel_std is not None:    config.near_vel_std = a.near_vel_std
    if a.capture_vel_w is not None:   config.capture_vel_w = a.capture_vel_w
    if a.no_capture_reward:           config.capture_reward = False
    if a.fixed_alpha is not None:     config.fixed_alpha = a.fixed_alpha
    if a.joint_damping is not None:   config.joint_damping = a.joint_damping
    if a.frame_skip is not None:      config.frame_skip = a.frame_skip
    if a.max_episode_steps is not None:
        config.max_episode_steps = a.max_episode_steps
    if a.init_from is not None:       config.init_from = a.init_from
    if a.no_swingup:                  config.swingup = False


def main():
    parse_args()
    os.makedirs(config.save_dir, exist_ok=True)
    wandb.init(project="triple-pendulum-RL", config=vars(config),
               name=f"sac-swingup-a{config.reset_angle_limit}-s{config.seed}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    # SAME_STEP: terminal/truncated obs lands in infos["final_obs"], next_obs is
    # already the autoreset state. we need final_obs for the buffer transition.
    envs = gym.vector.SyncVectorEnv(
        [make_env() for _ in range(config.n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    # instantiate models
    actor = Actor(obs_dim, act_dim, config.hidden_dim).to(device)
    q1 = QCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    q2 = QCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    q1_target = QCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    q2_target = QCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # optimizers
    actor_opt = optim.Adam(actor.parameters(), lr=config.lr)
    q_opt = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=config.lr)

    # auto tuned entropy temp alpha; target entropy = -act_dim (standard)
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=config.lr)
    if config.fixed_alpha > 0.0:
        with torch.no_grad():
            log_alpha.fill_(float(np.log(config.fixed_alpha)))
        print(f"alpha pinned at {config.fixed_alpha} (auto-tune disabled)")

    # warm start: load a previous actor. alpha is pinned low (last run
    # annealed to ~0.06) and the random warmup is skipped, otherwise the
    # high-entropy exploration phase erases the loaded behaviour before
    # alpha anneals back down.
    if config.init_from:
        sd = torch.load(config.init_from, map_location=device)
        actor.load_state_dict(sd)
        with torch.no_grad():
            log_alpha.fill_(float(np.log(0.05)))
        config.start_steps = 0
        print(f"warm-started actor from {config.init_from} (alpha init 0.05, no random warmup)")

    buffer = ReplayBuffer(obs_dim, act_dim, config.buffer_size)

    n_params = (sum(p.numel() for p in actor.parameters()) +
                sum(p.numel() for p in q1.parameters()) +
                sum(p.numel() for p in q2.parameters()))
    print(f"model has {n_params:,} parameters | n_envs={config.n_envs} device={device}")

    obs, _ = envs.reset(seed=config.seed)
    global_step = 0
    last_save = 0
    # windowed sps - total average was inflated by the random warmup
    sps_t, sps_step = time.time(), 0
    q_loss = actor_loss = alpha = None

    while global_step < config.total_timesteps:
        # 1. act (all envs at once)
        if global_step < config.start_steps:
            action = envs.action_space.sample() # random warmup
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device))
            action = a.cpu().numpy()

        next_obs, reward, terminations, truncations, infos = envs.step(action)

        # store real next state: on autoreset, next_obs is already the new ep,
        # the actual s' we transitioned into is infos["final_obs"]
        real_next = np.array(next_obs, dtype=np.float32, copy=True)
        if "final_obs" in infos:
            for i in range(config.n_envs):
                if infos["_final_obs"][i]:
                    real_next[i] = infos["final_obs"][i]

        # terminated only as done - truncated is not a real terminal state,
        # we still want to bootstrap through it
        buffer.add_batch(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32).reshape(-1, 1),
            real_next,
            np.asarray(terminations, dtype=np.float32).reshape(-1, 1),
        )
        obs = next_obs
        global_step += config.n_envs

        if "episode" in infos:
            mask = infos["_episode"]
            if mask.any():
                wandb.log({
                    "charts/episodic_return": float(infos["episode"]["r"][mask].mean()),
                    "charts/episodic_length": float(infos["episode"]["l"][mask].mean()),
                }, step=global_step)

        # 2 and 3. ONE update per vector step once warmup done
        # (old loop did a full update after every single env step -> ~40 sps)
        if global_step >= config.start_steps:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(config.batch_size, device)
            alpha = log_alpha.exp().detach()

            # critic updates
            with torch.no_grad():
                next_a, next_logp = actor.sample(b_next)
                q_next = torch.min(q1_target(b_next, next_a),
                                   q2_target(b_next, next_a))
                # entropy augmented TD target: reward + gamma * (minQ - alpha*logpi)
                target = b_rew + config.gamma * (1 - b_done) * (q_next - alpha * next_logp)

            q1_loss = F.mse_loss(q1(b_obs, b_act), target)
            q2_loss = F.mse_loss(q2(b_obs, b_act), target)
            q_loss = q1_loss + q2_loss
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            # actor update: maximize minQ(s, pi(s)) - alpha * logpi
            new_a, logp = actor.sample(b_obs)
            q_new = torch.min(q1(b_obs, new_a), q2(b_obs, new_a))
            actor_loss = (alpha * logp - q_new).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # alpha update: drive entropy toward target_entropy.
            # With --fixed-alpha the temperature stays pinned: auto-tuning on a
            # warm-started (low-entropy) actor pushes alpha up, and the action
            # jitter it creates is fatal for the capture/stabilize skill.
            if config.fixed_alpha <= 0.0:
                alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

            # soft target update
            with torch.no_grad():
                for p, pt in zip(q1.parameters(), q1_target.parameters()):
                    pt.data.mul_(1 - config.tau).add_(config.tau * p.data)
                for p, pt in zip(q2.parameters(), q2_target.parameters()):
                    pt.data.mul_(1 - config.tau).add_(config.tau * p.data)

        # windowed sps over the last ~5k env steps
        if global_step - sps_step >= 5000:
            sps = int((global_step - sps_step) / (time.time() - sps_t))
            sps_t, sps_step = time.time(), global_step
            log = {"charts/SPS": sps}
            if q_loss is not None:
                log.update({
                    "losses/q_loss": q_loss.item(),
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha.item(),
                })
            wandb.log(log, step=global_step)
            a_str = f"{alpha.item():.3f}" if alpha is not None else "warmup"
            print(f"step {global_step}/{config.total_timesteps} | sps {sps} | alpha {a_str}")

        if global_step - last_save >= 250_000:
            torch.save(actor.state_dict(), os.path.join(config.save_dir, "sac_swingup_latest.pt"))
            last_save = global_step

    torch.save(actor.state_dict(), os.path.join(config.save_dir, "sac_swingup_final.pt"))
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    main()
