#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import torch
import h5py
import glob
import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from main import *

name = 'area1_hdf5_2'
model_path = '/data/dell5/userdir/maotx/Lens/model/lens_001_20.cpt'
BaseDir = '/data/inspur_disk03/userdir/wangcx/BASS_stack/area1/'+name
OutDir = '/data/dell5/userdir/maotx/Lens/result/'+name
check_dir(OutDir)
fps = glob.glob(BaseDir + '/*.hdf5')
fps = [i.replace(BaseDir + '/', '') for i in fps]


class MyDataset(Dataset):
    """My dataset."""

    def __init__(self, root_dir, path, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root_dir
        self.path = path
        self.transform = transform
        self.fp = h5py.File(os.path.join(root_dir, path), 'r')
        self.keys = list(self.fp.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        image = self.fp[self.keys[idx]]
        g = self._center(image['g'][...].reshape(-1))
        r = self._center(image['r'][...].reshape(-1))
        z = self._center(image['z'][...].reshape(-1))
        image = np.c_[g, r, z].reshape(101, 101, 3)
        key = self.keys[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'key': key}

    def _center(self, x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / std


class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        imate = np.clip(image, -100, 100)
        image = torch.from_numpy(image)
        return image


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

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        return image


def eval(BaseDir, fps=[], OutDir=OutDir, model_path=model_path):
    preprocess = transforms.Compose([RandomCrop(84), ToTensor()])
    model, _ = get_model()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location='cpu')
    model, epoch = load(model_path, model)
    print('loading {}'.format(model_path), epoch)
    model.eval()
    for fp_num in tqdm.tqdm(range(len(fps))):
        fp = fps[fp_num]
        BASS_ds = MyDataset(root_dir=BaseDir, path=fp, transform=preprocess)
        BASS_dl = DataLoader(BASS_ds, batch_size=args.batch_size, num_workers=1)
        PROB = []
        with torch.no_grad():
            for data_step in BASS_dl:
                key = data_step['key']
                prob = torch.sigmoid(model(data_step['image'])).numpy()
                PROB.append(zip(key,prob))
            with open(os.path.join(OutDir, fp[:-5]+'.txt'),'w') as FP:
                for Item in PROB:
                    for P_key, P_value in Item:
                        temp = "{}\t{}\t{}\n".format(P_key, P_value[0], P_value[1])
                        FP.writelines(temp)


if __name__ == "__main__":
    eval(BaseDir, fps=fps, OutDir=OutDir, model_path=model_path)
