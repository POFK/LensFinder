#!/usr/bin/env python
# coding: utf-8
import numpy as np
import h5py

#__all__ = ['readdata', 'Map']


def preprocess(x, sigma=100):
    x -= x.min()
    x /= x.max()
    return x


def readdata(file, key):
    with h5py.File(file, 'r') as fp:
        image = fp[key]
        datar = image['r'][...]
        datag = image['g'][...]
        dataz = image['z'][...]
    datar = preprocess(datar)
    datag = preprocess(datag)
    dataz = preprocess(dataz)
    return datar, datag, dataz


class Map(object):

    """Docstring for Map. """

    def __init__(self, m, M, beta):
        """TODO: to be defined1. """
        self.min = m
        self.max = M
        self.beta = beta

    def color_F(self, x):
        return np.arcsinh(x/self.beta)

    def color_f(self, x):
        bool_min = x < self.min
        bool_max = x > self.max
        fx = self.color_F(x-self.min) / \
            self.color_F(self.max-self.min)
        fx[bool_min] = 0.
        fx[bool_max] = 1.
        return fx

    def color_map(self, r, g, z):
        I = (r+g+z)/3.
        bool_I = I == 0
        I[bool_I] = 1
        RGB = [c*self.color_f(I)/I for c in [r, g, z]]
        RGB[0][bool_I] = 0.
        RGB[1][bool_I] = 0.
        RGB[2][bool_I] = 0.
        RGB = np.stack(RGB)
        RGB_max = np.max(RGB, axis=0)
        RGB_max[RGB_max < 1] = 1.
        RGB /= RGB_max[None, :]*np.ones(RGB.shape)
        return RGB

    def __call__(self, r, g, z):
        return self.color_map(r, g, z)


if __name__ == "__main__":
    import glob
    import os
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    params = {
        'figure.figsize': [4.5, 4.5],
        'image.cmap': 'gray',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'xtick.major.top': False,
        'xtick.major.bottom': False,
        'ytick.major.left': False,
        'ytick.major.right': False,
    }
    plt.rcParams.update(params)

    DirBase = "/data/inspur_disk03/userdir/wangcx/BASS_stack/area2/DEV_COMP_hdf5"
    OutBase = ""
    data_shape = [101, 101, 3]
    fps = glob.glob(os.path.join(DirBase, '*.hdf5'))

    myshow = Map(0.15, 1, 0.3)
    """
    from myshow import Map, readdata
    """

    data = readdata(fps[0], '87599020487')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    im = myshow(data[0], data[1], data[2])
    axes[0].imshow(im[0])
    axes[1].imshow(im[1])
    axes[2].imshow(im[2])
    axes[3].imshow(im.transpose(1, 2, 0))
    fig.savefig('test.png')
    fig.show()
