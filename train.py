"""Train and evaluate capsule network on MNIST

Usage:
  train.py -r 3 -d 0.0005 -e 1 -b 128
  train.py -r 3 -d 0.0005 -e 1 -b 128
"""

import torch
from torch.nn import functional as F

from torchvision import datasets, transforms
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from tqdm import tqdm

from modules import CapsuleNetwork, Decoder


def train(load_path=None, save_path='capsnet', n_routing=3, decoder_loss=0, n_epochs=1, batch_size=128, log_interval=10, test_interval=100, plot_reconstruction=False, writer=None, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
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
        caps_net.load_state_dict(torch.load(load_path + f'_r{n_routing}_d{decoder_loss}'))

    params = caps_net.parameters()
    if decoder_loss > 0:
        decoder = Decoder()
        params = list(params) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params)

    caps_net.train()
    for epoch in tqdm(range(n_epochs), unit='epoch'):
        pbar = tqdm(enumerate(train_loader), unit='batch')
        for batch_idx, (data, target) in pbar:
            n_iter = epoch * len(train_loader) + batch_idx

            optimizer.zero_grad()
            outputs, vectors = caps_net(data)

            onehot = onehot_(target, n_categories=10)
            capsule_loss = margin_loss(outputs, onehot)
            loss = capsule_loss
            losses = dict(capsule=capsule_loss)
            if (decoder_loss > 0):
                decoder_input = vectors * onehot.unsqueeze(-1)
                decoder_input = decoder_input.flatten(1)
                reconstruction = decoder(decoder_input)
                reconstruction_loss = decoder_loss * F.mse_loss(reconstruction, data.flatten(1), reduction='sum')
                loss = capsule_loss + reconstruction_loss
                losses.update(reconstruction=reconstruction_loss)

            loss.backward()
            losses.update(joint=loss)
            writer.add_scalars('training/losses', losses, n_iter)

            optimizer.step()

            if (batch_idx % log_interval == 0):
                if plot_reconstruction:
                    import matplotlib.pyplot as plt
                    _, axes = plt.subplots(nrows=1, ncols=2,
                                           sharex=True, sharey=True,
                                           subplot_kw={'xticks': [],
                                                       'yticks': []})
                    axes[0].pcolor(data.flatten(1)[0].detach().numpy().reshape((28, 28)), cmap='Greys')
                    axes[1].pcolor(reconstruction[0].detach().numpy().reshape((28, 28)), cmap='Greys')
                    plt.savefig(save_path + f'{epoch}_{batch_idx}')
                    writer.add_image('digit', vutils.make_grid(data[0], normalize=True, scale_each=True), n_iter)
                    writer.add_image('reconstruction', vutils.make_grid(reconstruction[0].view(28, 28), normalize=True, scale_each=True), n_iter)

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} = {:.6f} + {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    capsule_loss, reconstruction_loss if decoder_loss > 0 else 0))

            if ((batch_idx+1) % test_interval == 0):
                test(caps_net, n_iter, batch_size, writer=writer, **kwargs)

    torch.save(caps_net.state_dict(), save_path + f'_r{n_routing}_d{decoder_loss}')

    return caps_net


def test(model, n_iter=0, batch_size=128, writer=None, **kwargs):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data',
                       train=False,
                       download=True,
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
            output, _ = model(data)
            onehot = onehot_(target, n_categories=10)
            test_loss += margin_loss(output, onehot, reduce='sum').item()
            pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
            correct += (pred == target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    error_rate = 100 - (100. * correct / len(test_loader.dataset))

    print('\nTest set: Avg loss: {:.4f}, Ac: {}/{} Error: {:.3f}\n'.format(
          test_loss, correct, len(test_loader.dataset),
          error_rate))

    writer.add_scalars('test/metrics', dict(error=error_rate, accuracy=correct/len(test_loader.dataset)), n_iter)


def evaluate(**kwargs):
    writer = SummaryWriter()
    caps_net = train(writer=writer, **kwargs)
    test(caps_net, writer=writer, **kwargs)
    writer.close()


def onehot_(categorical_labels, n_categories):
    categorical_labels = categorical_labels.unsqueeze(-1)
    onehot = torch.zeros(categorical_labels.shape[0], n_categories)
    onehot = onehot.scatter_(1, categorical_labels, 1)
    return onehot


def margin_loss(probs, target, reduce='mean'):
    '''
    vectors: vectors whose scales represent the log prob.
    target: onehot categorical labels.
    '''
    M_P = 0.9
    M_M = 0.1
    LAMBDA = 0.5

    L = target * F.relu(M_P - probs)**2
    L += LAMBDA * (1. - target) * F.relu(probs - M_M)**2
    loss = L.sum(-1)

    if reduce == 'mean':
        return loss.mean()
    elif reduce == 'sum':
        return loss.sum()
    else:
        raise ValueError()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train and Evaluate capsule network on MNIST.")
    parser.add_argument('--save_path', '-s', help="Path to save the model after training.", default='capsnet')
    parser.add_argument('--load_path', '-m', help="Path to saved model to be trained/evaluated.", default=None)
    parser.add_argument('--n_routing', '-r', type=int, help="Number of routing iterations for routing-by-agreement mechanism.", default=3)
    parser.add_argument('--decoder_loss', '-d', type=float, help="Scale of decoder loss.", default=0)
    parser.add_argument('--n_epochs', '-e', type=int, help="Number of epochs.", default=3)
    parser.add_argument('--batch_size', '-b', type=int, help="Batch size for training and testing.", default=128)
    parser.add_argument('--log_interval', '-l', type=int, help="Number of batches to report the training accuracy and loss.", default=10)
    parser.add_argument('--plot_reconstruction', help="Flag to plot the reconstructions.", action='store_true')

    args = parser.parse_args()
    evaluate(**vars(args))
