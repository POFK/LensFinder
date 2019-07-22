#!/usr/bin/env python
# coding=utf-8
import glob
import os
import matplotlib.pyplot as plt

from myshow import Map, readdata

#------------------------------------------------------------
# set plot parameters
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
#---------------------- load data ---------------------------
DirBase = "/data/inspur_disk03/userdir/wangcx/BASS_stack/area2/DEV_COMP_hdf5"
OutBase = ""
data_shape = [101, 101, 3]
fps = glob.glob(os.path.join(DirBase, '*.hdf5'))
data = readdata(fps[0], '87599020487')
#----------------------  init  ------------------------------
myshow = Map(0.15, 1, 0.3)
"""
The Map class should be imported by `from myshow import Map, readdata`.
There are three parameters:
    m: the minimal value cut in the image, default 0.15
    M: the maximal value cut in the image, default 1
    beta: this value is used to scale the image, when x << beta, 
        a linear scale and x >> beta a log scale, default 0.3
you can test different parameters by yourself.
"""
#----------------------- plot -------------------------------
"""
You can plot the image by your self. Here, the function myshow is
used to scale the image to find more details. It return the scaled 
numpy array.
"""
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
im = myshow(data[0], data[1], data[2])
axes[0].imshow(im[0])
axes[1].imshow(im[1])
axes[2].imshow(im[2])
axes[3].imshow(im.transpose(1, 2, 0))
fig.savefig('test.png')
fig.show()
