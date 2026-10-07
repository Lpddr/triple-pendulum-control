"""One stationary neural policy; inference has no trajectory optimizer/controller."""
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(11,hidden),nn.Tanh(),nn.Linear(hidden,hidden),
                               nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.linear=nn.Linear(11,1,bias=False)
        self.register_buffer('scale',torch.tensor([1.,1,1,1,1,1,1,.2,.2,.2,.2]))

    def features(self,x):
        return torch.cat((x[...,:1],torch.sin(x[...,1:4]),torch.cos(x[...,1:4])-1,x[...,4:]),-1)*self.scale

    def forward(self,x):
        z=self.features(x)
        return self.net(z)-self.net(torch.zeros_like(z[:1]))+self.linear(z)

    def action(self,x):
        return self(x).clamp(-1,1)


class PhasePolicy(nn.Module):
    """A single phase-conditioned affine neural policy learned by policy search.

    Phase is episode age, saturated at the last embedding. Every step uses the
    exact same network expression. No online optimization or second controller.
    The reference buffer and embeddings are learned training artifacts.
    """
    def __init__(self, phases):
        super().__init__()
        self.embedding=nn.Embedding(phases,9,dtype=torch.float64)
        self.register_buffer('reference',torch.zeros((phases,8),dtype=torch.float64))

    def forward(self,x,step=0):
        index=torch.as_tensor(step,dtype=torch.long,device=x.device).clamp(0,self.embedding.num_embeddings-1)
        params=self.embedding(index)
        error=x.to(torch.float64)-self.reference[index]
        return (params[..., :8]*error).sum(-1,keepdim=True)+params[...,8:]

    def action(self,x,step=0):
        return self(x,step).clamp(-1,1)
