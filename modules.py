import math
import copy
import torch
from torch import nn
from torch import functional as F


class Decoder(nn.Module):
    """Decoder for the regularization by reconstruction loss."""
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class CapsuleNetwork(nn.Module):
    """Capsule Network for MNIST as described by Sabour et al. in arXiv:1710.09829v2."""
    def __init__(self, n_primary_caps=32, n_routing=3):
        super(CapsuleNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=9, stride=1),
            nn.ReLU())

        # 32x6x6 capsules with 8D output
        capsule = nn.Conv2d(256, 8, kernel_size=9, stride=2)
        self.prim_caps = nn.ModuleList(
            [copy.deepcopy(capsule) for _ in range(n_primary_caps)])

        # 10 capsules with 16D output
        self.digit_caps = CapsuleLayer(32*6*6, 8, 10, 16, n_routing=n_routing)

    def forward(self, x):
        x = self.layer1(x)

        # primary capsules
        us = []
        for caps in self.prim_caps:
            s = caps(x)
            v = squash(s)
            us.append(v)

        us = torch.stack(us, -1)

        # flatten CNN channels
        us = torch.flatten(us, 2, 4)
        vs = torch.transpose(us, -1, -2)  # [batch_size, n_primary_caps, n_primary_features]

        # digit capsules
        v = self.digit_caps(vs)

        classes = torch.norm(v, p=2, dim=-1)
        return classes, v


class CapsuleLayer(nn.Module):
    '''Capsule Layer with routing-by-agreement mechanism.'''
    def __init__(self, in_caps, in_features, out_caps, out_features, n_routing=3):
        super(CapsuleLayer, self).__init__()
        self.in_caps = in_caps
        self.in_features = in_features
        self.out_caps = out_caps
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(self.out_caps, self.in_caps, self.in_features, self.out_features),
                                 requires_grad=True)

        self.n_routing = n_routing
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, vs):
        '''vs: [batch_size, in_caps, in_features]'''
        vs = vs.unsqueeze(2)
        vs = vs.unsqueeze(1)

        # Unsqueeze first dimension to allow broadcasting over the batch.
        u_hat = vs.matmul(self.W.unsqueeze(0))
        u_hat = u_hat.squeeze(3)
        return self.routing(u_hat, self.n_routing)

    def routing(self, u, r):
        '''u: [batch_size, out_caps, in_caps, out_features]'''
        B = torch.zeros((self.out_caps, self.in_caps))
        for i in range(r):
            c = torch.softmax(B, dim=-1).unsqueeze(0).unsqueeze(2)

            s = c.matmul(u).squeeze(2)
            v = squash(s)

            B = B + torch.matmul(
                u.unsqueeze(-2),
                v.unsqueeze(2).unsqueeze(-1)).squeeze(-1).squeeze(-1).mean(0)

        return v


def squash(x):
    n_input2 = F.norm(x)
    out = n_input2 / (1 + n_input2)
    n_input = F.norm(x, p=1, dim=1, keepdim=True)
    out = out * x / n_input
    return out
