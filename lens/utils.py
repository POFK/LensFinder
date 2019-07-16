#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def get_device(use_cuda=True):
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def get_model_device(model, device):
    if device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)
    return model


def check_dir(path):
    if not os.path.isdir(path):
        print('mkdir: ', path)
        os.makedirs(path)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


"""
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for data_step in train_dl:
            xb, yb = data_step['image'].to(device), data_step['label'].to(
                device)
            train_loss, _ = loss_batch(model, loss_func, xb, yb, opt)
       #writer.add_scalar('loss/train', train_loss, global_step=epoch)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[
                loss_batch(model, loss_func, data_step['image'].to(device),
                           data_step['label'].to(device))
                for data_step in valid_dl
            ])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
       #writer.add_scalar('loss/valid', val_loss, global_step=epoch)
        print(epoch, val_loss)
"""
