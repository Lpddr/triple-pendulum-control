import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # init orthogonal weights - keeps activations from exploding or vanishing
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        # critic: obs -> 256 -> 256 -> 256 -> 1 (3 hidden layers, 1 output)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0), # value head, gain 1.0
        )

        # actor: obs -> 256 -> 256 -> 256 -> act_dim (3 hidden layers, act_dim output)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01), # policy head, gain 0.01
        )

        # state independent log std, init 0 -> std = 1.0
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x).squeeze(-1) # remove last dim, output shape: (batch,)
    
    def get_action_and_value(self, x, action=None):
        # get mean from actor network
        mu = self.actor(x)
        # get std from log std parameter
        std = torch.exp(self.actor_logstd.expand_as(mu))
        # create normal distribution with mean and std
        dist = Normal(mu, std)
        # if action is not provided, sample from the distribution
        if action is None:
            action = dist.sample()
        # calculate log probability of the action
        log_prob = dist.log_prob(action).sum(-1)
        # calculate entropy of the dist
        entropy = dist.entropy().sum(-1)
        # get value from the critic network
        value = self.get_value(x)
        return action, log_prob, entropy, value

