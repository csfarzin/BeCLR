import argparse
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from plot.tsne import tSNE
from data_aug.transformation import transform
from model import Model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'runs/May24_15-11-04_gpuserver.stat.uni-muenchen.de/cifar10_K100_resnet18_0.001_128_0.1_200_512_300_model.pth'
model = Model(base_model="resnet18",
                 fc_dim=128,
                 k_subs=100,
                 layer_sizes=[64, 1],
                 use_bn=False,
                 dr_rate=0.2)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.to(device)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
train_transform, test_transform = transform(32, mean, std)

test_data = CIFAR10(
    root='data', train=False, transform=test_transform, download=True)
#test_data, _ = torch.utils.data.random_split(test_data, [5000, 5000])

test_loader = DataLoader(
    test_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)


train_data = CIFAR10(
    root='data', train=True, transform=test_transform, download=True)
train_data, _ = torch.utils.data.random_split(train_data, [5000, 45000])

train_loader = DataLoader(
    train_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

feature_bank = []
feature_labels = []
for data, target in tqdm(train_loader):
    fc_out, out = model(data.to(device))
    feature_bank.append(fc_out.detach().cpu())
    feature_labels.append(target)
    
feature_bank = torch.cat(feature_bank, dim=0)
feature_labels = torch.cat(feature_labels, dim=0)


plott = tSNE(2, feature_bank.numpy(), 'cifar10_k100_300_train')
plott.tsne_plt(feature_labels.numpy())