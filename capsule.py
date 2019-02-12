import torch
from torch import nn
from torch import functional as F


class CapsuleLayer(nn.Module):
    '''Capsule Layer with routing-by-agreement mechanism.'''
    def __init__(self, in_caps, in_features, out_caps, out_features, n_routing=3):
        super(CapsuleLayer, self).__init__()
        self.in_caps = in_caps
        self.in_features = in_features
        self.out_caps = out_caps
        self.out_features = out_features

        self.W = torch.randn((self.out_caps, self.in_caps, self.in_features, self.out_features),
                             requires_grad=True)

        self.n_routing = n_routing

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
