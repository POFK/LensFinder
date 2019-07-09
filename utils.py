#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_data(train_ds, valid_ds, bs, bs_test, num_workers=0):
    return (
        DataLoader(train_ds,
                   batch_size=bs,
                   shuffle=True,
                   num_workers=num_workers,
                   drop_last=True),
        DataLoader(valid_ds, batch_size=bs_test),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


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
