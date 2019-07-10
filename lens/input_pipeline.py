#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

__all__ = ['MyDataset', 'ToTensor', 'RandomCrop', 'get_data']


def get_data(train_ds, valid_ds, bs, bs_test, num_workers=0):
    return (
        DataLoader(train_ds,
                   batch_size=bs,
                   shuffle=True,
                   num_workers=num_workers,
                   drop_last=True),
        DataLoader(valid_ds, batch_size=bs_test),
    )


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
    def __init__(self, num_class=2):
        self.num_class = num_class

    def __call__(self, sample):
        num_class = self.num_class
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
