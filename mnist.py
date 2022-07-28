import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

BATCH_SIZE = 2048
EPOCHS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_train = torchvision.datasets.MNIST(root='./dataset_method_1', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]), download=True, )
dataset_test = torchvision.datasets.MNIST(root='./dataset_method_1', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]), download=False)
data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False)
ce = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, curr_epoch, LR, min_lr, epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if curr_epoch < warmup_epochs:
        lr = LR * curr_epoch / warmup_epochs
    else:
        lr = min_lr + (LR - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (curr_epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet.resnet18(pretrained=True)
        self.cls = nn.Linear(self.backbone.fc.in_features, 10)
        self.backbone.fc = self.cls
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.resnet(x)
        # x = self.mlp(x)
        return x


model = net().to(DEVICE)
scaler = GradScaler()

optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)
ep_pbar = tqdm(range(EPOCHS))
LOG_DIR = './log'
import os

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
writer = SummaryWriter(log_dir=LOG_DIR)
train_iter = 0
for epoch in ep_pbar:
    train_loss, test_loss = [], []
    model.train()
    data_iter_step = 0
    pbar = tqdm(data_loader_train, total=len(data_loader_train))
    for data, target in pbar:
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, epochs=EPOCHS, LR=1.5e-3,
                             min_lr=0, warmup_epochs=40)
        writer.add_scalar('learning_rate/iter', optimizer.param_groups[0]["lr"], train_iter)
        train_iter += 1
        data_iter_step += 1
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        with autocast():
            y = model(data)

        loss = ce(y, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(loss.cpu().detach())

        pbar.set_postfix({'Epoch': epoch,
                          'loss': float(np.mean(train_loss)),
                          'lr': optimizer.param_groups[0]["lr"]})

    model.eval()
    pbar = tqdm(data_loader_test, total=len(data_loader_test))
    for data, target in pbar:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        with autocast():
            y = model(data)
        loss = ce(y, target)
        test_loss.append(loss.cpu().detach())
    # lr_sch.step()
    ep_pbar.set_postfix({'Epoch': epoch,
                         'loss': float(np.mean(train_loss)),
                         'test_loss': float(np.mean(test_loss)),

                         'lr': optimizer.param_groups[0]["lr"]})

    writer.add_scalar('Loss/train', float(np.mean(train_loss)), epoch)
    writer.add_scalar('Loss/test', float(np.mean(test_loss)), epoch)
    writer.add_scalar('learning_rate/epoch', optimizer.param_groups[0]["lr"], epoch)
    writer.close()
