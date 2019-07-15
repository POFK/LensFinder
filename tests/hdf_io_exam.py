#!/usr/bin/env python
# coding=utf-8
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset
import os


class HdfDataset(Dataset):

    def __init__(self, path, num_workers=1):
        self.path = path
        self.num_workers = num_workers
        self.fp = h5py.File(self.path, 'r')
        self.keys = list(self.fp.keys())
        if num_workers > 1:
            self.fp.close()

    def _init_fn(self, num):
        self.fp = h5py.File(self.path, 'r')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.fp[key]
        data_key = data['key'][...]
        data_val = data['value'][...]
        return {'value': data_val, 'key': data_key, 'k': key}


class HdfConcatDataset(ConcatDataset):
    def _init_fn(self, num):
        for ds in self.datasets:
            ds._init_fn(num)
