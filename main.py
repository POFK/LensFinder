#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from ResNet import resnet14 as resnet
from utils import get_data, loss_batch
from checkpoint import save, load
from flags import args
#============================================================
#DataDir = "/data/storage1/LensFinder/data_otherlens"
DataDir = "/data/storage1/LensFinder/data_old"
log_dir = os.path.join(args.base_dir, args.log_dir + '/' + args.name)
model_dir = os.path.join(args.base_dir, args.model_dir + '/' + args.name)
path_tr = os.path.join(DataDir, "training.npy")
path_va = os.path.join(DataDir, "valid.npy")


def check_dir(path):
    if not os.path.isdir(path):
        print('mkdir: ', path)
        os.makedirs(path)


check_dir(log_dir)
check_dir(model_dir)
#============================================================
device = torch.device("cuda:0" if args.use_cuda else "cpu")
num_class = 2


### data loader
class MyDataset(Dataset):
    """My dataset."""

    def __init__(self, path, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        data = np.load(path)
        self.data = {
            'image': data['image'],
            'label': data['label'].reshape(-1, 1)
        }
        self.transform = transform

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        sample = {
            'image': self.data['image'][idx],
            'label': self.data['label'][idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        imate = np.clip(image, -100, 100)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)
        label = F.one_hot(label, num_classes=num_class).view(num_class)
        return {'image': image, 'label': label}


class RandomCrop(object):
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
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        return {'image': image, 'label': label}


#============================================================
def get_model():
    model = resnet(input_channels=3, num_classes=2)
    #   print(model)
    #print(model.state_dict().keys())
    if device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)
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
