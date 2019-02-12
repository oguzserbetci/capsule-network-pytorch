import copy
import torch
from torch import nn
from capsule import CapsuleLayer
from capsule import squash


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
        self.digit_caps = CapsuleLayer(1152, 8, 10, 16, n_routing=n_routing)

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
        # vs: [batch_size, n_primary_caps, n_primary_features]
        vs = torch.transpose(us, -1, -2)

        # digit capsules
        v = self.digit_caps(vs)

        return v


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
