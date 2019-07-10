#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from input_pipeline import *
from ResNet import resnet14 as resnet
from utils import loss_batch, check_dir, get_model_device, get_device
from checkpoint import save, load
from flags import args

__all__ = [
    'get_model', 'args', 'save', 'load', 'resnet', 'get_data', 'loss_batch', 'check_dir'
]

# ============================================================
#DataDir = "/data/storage1/LensFinder/data_otherlens"
DataDir = "/data/storage1/LensFinder/data_old"
log_dir = os.path.join(args.base_dir, args.log_dir + '/' + args.name)
model_dir = os.path.join(args.base_dir, args.model_dir + '/' + args.name)
path_tr = os.path.join(DataDir, "training.npy")
path_va = os.path.join(DataDir, "valid.npy")


# ============================================================
device = torch.device("cuda:0" if args.use_cuda else "cpu")
num_class = 2


def get_model():
    model = resnet(input_channels=3, num_classes=2)
    #   print(model)
    # print(model.state_dict().keys())
    model = get_model_device(model, device)
    return model, torch.optim.Adam(model.parameters(),
                                   lr=args.lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=args.weight_decay,
                                   amsgrad=False)


def loss_func(x, y):
    x = torch.sigmoid(x)
    y = y.type(torch.float32)
    return F.binary_cross_entropy(x, y)


def _train(epochs,
           model,
           loss_func,
           opt,
           train_dl,
           valid_dl,
           writer,
           epoch_old=0):
    for epoch in range(epoch_old, epoch_old + epochs):
        model.train()
        for data_step in train_dl:
            xb, yb = data_step['image'].to(device), data_step['label'].to(
                device)
            train_loss, _ = loss_batch(model, loss_func, xb, yb, opt)
        writer.add_scalar('loss/train', train_loss, global_step=epoch)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[
                loss_batch(model, loss_func, data_step['image'].to(device),
                           data_step['label'].to(device))
                for data_step in valid_dl
            ])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        writer.add_scalar('loss/valid', val_loss, global_step=epoch)
        print(epoch, val_loss)
        model_path = os.path.join(model_dir,
                                  "{}_{}.cpt".format(args.name, epoch))
        save(model_path, epoch, model, val_loss, device, opt=opt)


def train(model_path=None):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    preprocess = transforms.Compose([RandomCrop(84), ToTensor()])
    train_ds = MyDataset(path=path_tr, transform=preprocess)
    valid_ds = MyDataset(path=path_va, transform=preprocess)
    train_dl, valid_dl = get_data(train_ds,
                                  valid_ds,
                                  bs=args.batch_size,
                                  bs_test=args.test_batch_size)
    model, opt = get_model()
    if load is not None:
        model, epoch = load(model_path, model, opt=opt)
    _train(args.epochs, model, loss_func, opt, train_dl, valid_dl, writer)
    writer.close()


def eval_test(name, epoch):
    path_te = os.path.join(DataDir, "test.npy")
    preprocess = transforms.Compose([RandomCrop(84), ToTensor()])
    test_ds = MyDataset(path=path_te, transform=preprocess)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    model, _ = get_model()
    model_path = os.path.join(model_dir, "{}_{}.cpt".format(name, epoch))
    model, epoch = load(model_path, model)
    print('loading {}'.format(model_path), epoch)
    model.eval()
    PROB = []
    with torch.no_grad():
        for data_step in test_dl:
            prob = torch.sigmoid(model(data_step['image'].to(device)))
            PROB.append(prob.cpu().numpy())
    PROB = np.vstack(PROB)
    return PROB


if __name__ == "__main__":
    check_dir(log_dir)
    check_dir(model_dir)
    if args.mode == 'eval':
        probability = eval_test(args.name, args.epoch)
        label = np.load(os.path.join(DataDir, "test.npy"))['label']
        np.save('eval_test.npy', np.c_[probability, label])
    elif args.mode == 'train':
        if args.epoch == 0:
            train()
        else:
            model_path = os.path.join(
                model_dir, "{}_{}.cpt".format(args.name, args.epoch))
            train(model_path=model_path)
