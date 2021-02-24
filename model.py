import torch
import torch.nn as nn
from torchvision import models


nclasses = 20

# Resnext model
resnext = models.resnext101_32x8d(pretrained=True)
for param in resnext.parameters():
    param.requires_grad = False

resnext.fc = nn.Sequential(
    nn.Linear(2048, nclasses),
    torch.nn.LogSoftmax(dim = 1))

# DeiT-tiny model
deit_tiny = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
for param in deit_tiny.parameters():
    param.requires_grad = False

deit_tiny.head = nn.Sequential(
    nn.Linear(192, nclasses),
    torch.nn.LogSoftmax(dim = 1))

# DeiT-base model
deit_base = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
for param in deit_base.parameters():
    param.requires_grad = False

deit_base.head = nn.Sequential(
    nn.Linear(768, nclasses),
    torch.nn.LogSoftmax(dim = 1))


model_dict = {'deit_tiny': deit_tiny,
              'deit_base': deit_base,
              'resnext': resnext}