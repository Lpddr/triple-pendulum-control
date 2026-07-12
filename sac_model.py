"""SAC (soft actor critic) networks: squashed gaussian actor + twin Q critics
differences from ppo: 
- actor ouputs mean and log_std per state (state dependent std; PPo used
one global std). the std is how SAD modulates exploration per state
- actions are squashed through tanh -> [-1, 1], with log_prob correction
- critics are Q(s, a): they take state and action, output one scalar each.

SAC trains the actor through the critic's opinion of differentiable actions, 
using entropy as a first class objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5, 2 # clamp for numerical stability

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
        ) # ^^ (above) what matters about this state
        self.mean_head = nn.Linear(hidden_dim, act_dim) # what action should i take
        self.logstd_head = nn.Linear(hidden_dim, act_dim) # how sure am i about the action

    def forward(self, obs):
        # computes raw gaussian params and applies the clamp
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.logstd_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs):
        # sample a squashed action + its log-prob (used everywehre in training)
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample() # reparameterized sample -> gradients flow through
        # ^ computes the sample as mean + std * epsilon where epsilon is noise drawn seperately
        # sample is now a differentiable function of mean and std, so gradients flow through it
        action = torch.tanh(x) # squash to [-1, 1]
        # log prob of the squashed action: gaussian logprob - tanh correction
        logprob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        logprob = logprob.sum(-1, keepdim=True)
        return action, logprob
    
    def deterministic(self, obs):
        # mean action for evaluation/watching, no sampling
        mean, _ = self(obs)
        return torch.tanh(mean)
    
class QCritic(nn.Module):
    # Q(s, a): concatenates state and action, outputs a scalar
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=1))
    
