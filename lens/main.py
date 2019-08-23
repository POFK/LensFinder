#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
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
#DataDir = "/data/storage1/LensFinder/0712/data"
DataDir = "/data/storage1/LensFinder/0822/data"
log_dir = os.path.join(args.base_dir, args.log_dir + '/' + args.name)
model_dir = os.path.join(args.base_dir, args.model_dir + '/' + args.name)
path_tr = os.path.join(DataDir, "training.npy")
path_va = os.path.join(DataDir, "valid.npy")
# ============================================================
device = get_device(args.use_cuda)
num_class = args.num_class
random_crop = args.crop_range


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
           epoch_old=1):
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
    preprocess = transforms.Compose([RandomCrop(random_crop), ToTensor()])
    train_ds = MyDataset(path=path_tr, transform=preprocess)
    valid_ds = MyDataset(path=path_va, transform=preprocess)
    train_dl, valid_dl = get_data(train_ds,
                                  valid_ds,
                                  bs=args.batch_size,
                                  bs_test=args.test_batch_size)
    model, opt = get_model()
    epoch_big = 1
    if model_path is not None:
        model, epoch_big = load(model_path, model, opt=opt)
    _train(args.epochs, model, loss_func, opt, train_dl,
           valid_dl, writer, epoch_old=epoch_big)
    writer.close()

class Crop_test(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = (h - new_h)//2
        left = (w - new_w)//2
        image = image[top:top + new_h, left:left + new_w]
        return {'image': image, 'label': label}

def eval_test(name, epoch):
    path_te = os.path.join(DataDir, "test.npy")
    preprocess = transforms.Compose([Crop_test(random_crop), ToTensor()])
    test_ds = MyDataset(path=path_te, transform=preprocess)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    model, _ = get_model()
    # modify the save function later to fix this bug
    if device.type == 'cpu':
        model = torch.nn.DataParallel(model)
    # modify the save function later to fix this bug
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
    print(args)
    if args.mode == 'eval':
        probability = eval_test(args.name, args.epoch)
        label = np.load(os.path.join(DataDir, "test.npy"))['label']
        fp_save = os.path.join(model_dir, 'eval_test_{}.npy'.format(args.epoch))
        np.save(fp_save, np.c_[probability, label])
    elif args.mode == 'train':
        if args.epoch == 0:
            train()
        else:
            model_path = os.path.join(
                model_dir, "{}_{}.cpt".format(args.name, args.epoch))
            print("load model {}".format(model_path))
            train(model_path=model_path)
