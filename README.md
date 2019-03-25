# Capsule Network [1] for MNIST, 

Pytorch implementation for Capsule Network. Setup python environment and run training with:

```shell
pipenv install
python train.py -r 3 -d 0.0005 -e 1 -b 128
```

Detailed information on training and generated images from the decoder can be seen using tensorboard:

```shell
tensorboard --logdir runs
```

[1]: Sabour et. al [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

