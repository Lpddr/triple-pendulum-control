"""SAC training for trip pend.
structure: 1. act in env, store transition -> 2. sample minibatch from
replay buffer -> 3. update critics, actor, alpha -> repeat.
no rollout phases, no gae, no clipping - a complete different loop than ppo

vectorized: n_envs step in parallel, but only ONE gradient update per
vector step (not per env). that was the big sps killer before - full
critic/actor/alpha update after every single mujoco step on cpu.
"""

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
from sac_model import Actor, QCritic

@dataclass
class cfg:
    env_id: str = "InvertedTriplePendulum-v0"
    reset_noise_scale: float = 0.05
    total_timesteps: int = 1_000_000 # sac is sample efficient, start 10x smaller than ppo
    n_envs: int = 16 # same as ppo - free env steps while the update still dominates
    buffer_size: int = 1_000_000
    batch_size: int = 256
    start_steps: int = 5_000 # pure random warmup to seed the buffer
    gamma: float = 0.99
    tau: float = 0.005 # soft target update rate
    lr: float = 3e-4
    hidden_dim: int = 256
    seed: int = 1
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
        return gym.make(config.env_id, reset_noise_scale=config.reset_noise_scale)
    return thunk

def main():
    wandb.init(project="triple-pendulum-RL", config=vars(config), name="sac-trip-pend")
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

            # alpha update: drive entropy toward target_entropy
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
            torch.save(actor.state_dict(), "sac_triple_latest.pt")
            last_save = global_step

    torch.save(actor.state_dict(), "sac_triple_final.pt")
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    main()
