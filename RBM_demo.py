"""
    Copyright © 2022 Melrose-Lbt
    All rights reserved
    Filename: RBM_demo.py
    Description: Demo of how to use RBM.
    Created by Melrose-Lbt 2022-11-1
"""

from RestrictedBoltzmanMachine import RBM
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

batch_size_train = 32
batch_size_test = 32

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=batch_size_test, shuffle=True)


rbm = RBM(784, 500, 5, 32, 1, lr=0.01)
rbm.train(train_loader, test_loader)