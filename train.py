"""Train and evaluate capsule network on MNIST

Usage:
  train.py -r 3 -d 0.0005 -e 1 -b 128
  train.py -r 3 -d 0.0005 -e 1 -b 128
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch import nn
from torch import functional as F

from model import CapsuleNetwork, Decoder

def train(load_path=None, save_path='', n_routing=3, decoder_loss=0, n_epochs=1, batch_size=128, log_interval=10, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            'data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=batch_size,
        shuffle=True)

    caps_net = CapsuleNetwork(n_routing=n_routing)
    if load_path is not None:
        caps_net.load_state_dict(torch.load(load_path))
    if decoder_loss > 0:
        decoder = Decoder()

    optimizer = torch.optim.Adam(caps_net.parameters())
    caps_net.train()
    losses = []
    for epoch in range(n_epochs):
        losses.append([])
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = caps_net(data)

            onehot = onehot_(target, n_categories=10)
            loss = margin_loss(output, onehot)
            if (decoder_loss > 0):
                decoder_input = output * onehot.unsqueeze(-1)
                decoder_input = decoder_input.flatten(1)
                reconstruction = decoder(decoder_input)
                reconstruction_loss = torch.pairwise_distance(reconstruction, data.flatten(1), p=2).sum()
                loss = loss + decoder_loss * reconstruction_loss
            loss.backward()
            optimizer.step()
            losses[-1].append(loss.item())
            if (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    torch.save(caps_net.state_dict(), save_path + f'_r{n_routing}_d{decoder_loss}')

    return caps_net


def test():
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            'data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=batch_size,
        shuffle=True)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += margin_loss(output,
                                     target).item()  # sum up batch loss
            pred = output.norm(
                p=1, dim=-1).argmax(
                    dim=-1,
                    keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Avg loss: {:.4f}, Acc: {}/{} Error: {:.3f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 - (100. * correct / len(test_loader.dataset))))


def onehot_(categorical_labels, n_categories):
    categorical_labels = categorical_labels.unsqueeze(-1)
    onehot = torch.zeros(categorical_labels.shape[0], n_categories)
    onehot = onehot.scatter_(1, categorical_labels, 1)
    return onehot


def margin_loss(vectors, target, reduce='mean'):
    '''
    vectors: vectors whose scales represent the log prob.
    target: onehot categorical labels.
    '''
    ZERO = torch.Tensor([0.])

    M_P = 0.9
    M_M = 0.1
    LAMBDA = 0.5

    prob = F.norm(vectors, p=1, dim=-1)

    L = target * torch.max(ZERO, M_P - prob)**2
    L += LAMBDA * (1. - target) * torch.max(
        ZERO,
        prob - M_M)**2
    loss = L.sum(-1)

    if reduce == 'mean':
        return loss.mean()
    elif reduce == 'sum':
        return loss.sum()
    else:
        raise ArgumentError


def evaluate(**kwargs):
    caps_net = train(**kwargs)
    test(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train and Evaluate capsule network on MNIST.")
    parser.add_argument('--save_path', '-s', help="Path to save the model after training.", default='')
    parser.add_argument('--load_path', '-m', help="Path to saved model to be trained/evaluated.", default=None)
    parser.add_argument('--n_routing', '-r', type=int, help="Number of routing iterations for routing-by-agreement mechanism.", default=3)
    parser.add_argument('--decoder_loss', '-d', type=float, help="Scale of decoder loss.", default=0)
    parser.add_argument('--n_epochs', '-e', type=int, help="Number of epochs.", default=3)
    parser.add_argument('--batch_size', '-b', type=int, help="Batch size for training and testing.", default=128)
    parser.add_argument('--log_interval', '-l', type=int, help="Number of batches to report the training accuracy and loss.", default=10)

    args = parser.parse_args()
    evaluate(**vars(args))
