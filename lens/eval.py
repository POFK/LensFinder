#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import torch
import h5py
import tqdm

import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

from main import *

#name = 'area1_hdf5_2'
#name = 'area2_hdf5'
name = 'DEV_COMP_hdf5'
model_path = '/data/dell5/userdir/maotx/Lens/model/lens_049_45.cpt'
BaseDir = '/data/inspur_disk03/userdir/wangcx/BASS_stack/area2/'+name
OutDir = '/data/dell5/userdir/maotx/Lens/result/{}_{}'.format(
    name, model_path.split('/')[-1][:-4])
check_dir(OutDir)
fps = glob.glob(BaseDir + '/*.hdf5')
fps = [i.replace(BaseDir + '/', '') for i in fps]

class HdfDataset(Dataset):
    """My dataset."""

    def __init__(self, root, path, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.path = path
        self.transform = transform
        with h5py.File(os.path.join(root, path), 'r') as fp:
            self.keys = list(fp.keys())

    def _init_fn(self, num):
        self.fp = h5py.File(os.path.join(self.root, self.path), 'r')

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
        return {'image': image, 'key': key, 'path': self.path}

    def _center(self, x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / std


class HdfConcatDataset(ConcatDataset):

    def _init_fn(self, num):
        for ds in self.datasets:
            ds._init_fn(num)


class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        image = np.clip(image, -100, 100)
        image = torch.from_numpy(image)
        return image


class Crop(object):
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
        top = (h - new_h)//2
        left = (w - new_w)//2
        image = image[top:top + new_h, left:left + new_w]
        return image


def eval(BaseDir, fps=[], OutDir=OutDir, model_path=model_path):
    preprocess = transforms.Compose([Crop(84), ToTensor()])
    model, _ = get_model()
    model = torch.nn.DataParallel(model)
    _ = torch.load(model_path, map_location='cpu')
    model, epoch = load(model_path, model)
    print('loading {}'.format(model_path), epoch)
    model.eval()
    for fp in tqdm.tqdm(fps):
        BASS_ds = HdfDataset(root=BaseDir, path=fp, transform=preprocess)
        BASS_dl = DataLoader(BASS_ds, batch_size=args.batch_size,
                             num_workers=4, worker_init_fn=BASS_ds._init_fn)
        PROB = []
        with torch.no_grad():
            for data_step in BASS_dl:
                key = data_step['key']
                prob = torch.sigmoid(model(data_step['image'])).numpy()
                PROB.append(zip(key, prob))

        wpath = os.path.join(OutDir, fp.replace('.hdf5','.txt'))
        with open(wpath, 'w') as FP:
            for result in PROB:
                for key, P_value in result:
                    temp = "{}\t{}\t{}\n".format(key, P_value[0], P_value[1])
                    FP.writelines(temp)

if __name__ == "__main__":
    eval(BaseDir, fps=fps, OutDir=OutDir, model_path=model_path)

