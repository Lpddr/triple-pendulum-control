""" ppo training for my custom triple pendulum environment, 
uses InvertedTriplePendulum-v0 for the registered env. four phases
collect -> GAE -> update -> repeat
"""

import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AutoresetMode
import glfw
import wandb
import triple_pendulum # registers InvertedTriplePendulum-v0 env
from model import ActorCritic

# config
@dataclass
class Config:
    env_id: str = "InvertedTriplePendulum-v0"
    reset_noise_scale: float = 0.05 # 0.05 is easier to learn per readme
    total_timesteps: int = 10_000_000
    n_envs: int = 16
    n_steps: int = 512 # rollout = 16 * 512 = 8192 samples

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    n_minibatches: int = 32 # minibatch = 8192 / 32 = 256 samples
    target_kl: float = 0.02

    hidden_dim: int = 256
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    watch_after_training: bool = True # open the viewer when training finishes

config = Config()

def make_env():
    def thunk():
        return gym.make(config.env_id, reset_noise_scale=config.reset_noise_scale)
    return thunk

def watch(agent, device):
    # open mujoco windo and run the trained policy until the window is closed
    env = gym.make(config.env_id, render_mode="human")
    agent.eval()
    obs, _ = env.reset()
    env.render() # force the window to be created to watch its close flag
    renderer = env.unwrapped.mujoco_renderer
    try:
        while True:
            viewer = renderer.viewer
            # windown closed -> exit cleanly before next render
            if viewer is not None and glfw.window_should_close(viewer.window):
                break
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # use the policy mean, not a sample -> smooth, deterministic balancing
                action = agent.actor(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            obs, reward, term, trunc, _ = env.step(action) # auto renders in human mode
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()

def main():
    # init wandb
    wandb.init(project="triple-pendulum-RL", config=vars(config))
    # set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    # SAME_STEP autoreset restores the pre-1.0 behavior this loop assumes:
    # resets happen in-step and the terminal obs lands in infos["final_obs"].
    envs = gym.vector.SyncVectorEnv(
        [make_env() for _ in range(config.n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs) # episode return/length logging
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.array(envs.single_action_space.shape).prod())

    agent = ActorCritic(obs_dim, act_dim, config.hidden_dim).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"model has {n_params:,} parameters")
    optimizer = optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    batch_size = config.n_envs * config.n_steps
    minibatch_size = batch_size // config.n_minibatches
    n_updates = config.total_timesteps // batch_size

    # rollout storage
    obs = torch.zeros((config.n_steps, config.n_envs, obs_dim), device=device)
    actions = torch.zeros((config.n_steps, config.n_envs, act_dim), device=device)
    logprobs = torch.zeros((config.n_steps, config.n_envs), device=device)
    rewards = torch.zeros((config.n_steps, config.n_envs), device=device)
    dones = torch.zeros((config.n_steps, config.n_envs), device=device)
    values = torch.zeros((config.n_steps, config.n_envs), device=device)

    global_step = 0
    start = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(config.n_envs, device=device)

    for update in range(1, n_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / n_updates
            optimizer.param_groups[0]['lr'] = frac * config.lr

        # 1. collect a rollout
        for step in range(config.n_steps):
            global_step += config.n_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)

            # truncation bootstrap
            # env has 1000-step time limit, so once the policy can balance it
            # will truncate constantly. A truncated episode isn't a real terminal
            # state the pole was still up - so we bootstrap V(final_obs) into the
            # reward. Without this, PPO would learn that surviving to step 1000 is
            # worthless and undershoot. (Genuine terminations - the pole falling -
            # get no bootstrap, which is correct.)
            if "final_obs" in infos:
                final_obs = infos["final_obs"]
                final_mask = infos["_final_obs"]
                for i in range(config.n_envs):
                    if final_mask[i] and truncations[i] and not terminations[i]:
                        fo = final_obs[i] # fo = final observation
                        with torch.no_grad():
                            fv = agent.get_value(
                                torch.tensor(fo, dtype=torch.float32, device=device).unsqueeze(0)
                            ).item()
                        rewards[step, i] += config.gamma * fv

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "episode" in infos:
                mask = infos["_episode"]
                if mask.any():
                    wandb.log({
                        "charts/episodic_return": infos["episode"]["r"][mask].mean(),
                        "charts/episodic_length": infos["episode"]["l"][mask].mean(),
                    }, step = global_step)

        # 2. GAE
        # next_done marks episode boundaries (term or trunc), so advantage doesn't
        # leak across the resets. Truncation was already handled in the reward bootstrap above.
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_val = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_val = values[t + 1]
                delta = rewards[t] + config.gamma * next_val * next_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, act_dim)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 3. update
        idx = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):
            np.random.shuffle(idx)
            for start_i in range(0, batch_size, minibatch_size):
                mb = idx[start_i:start_i + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_unclipped = (newvalue - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(
                    newvalue - b_values[mb], -config.clip_coef, config.clip_coef
                )
                v_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        # log update
        sps = int(global_step / (time.time() - start))
        wandb.log({
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "charts/SPS": sps,
        }, step=global_step)
        print(f"update {update}/{n_updates}, step {global_step}, SPS {sps}, kl {approx_kl.item():.4f}")

        if update % 50 == 0:
            torch.save(agent.state_dict(), f"ppo_{config.env_id}_{global_step}.pt")

    envs.close()
    wandb.finish()

    if config.watch_after_training:
        watch(agent, device)

if __name__ == "__main__":
    main()