#!/usr/bin/env python
# coding=utf-8
import h5py
import os
import numpy as np
import unittest
from unittest import mock
from torch.utils.data import Dataset, DataLoader

from tests.hdf_io_exam import HdfDataset, HdfConcatDataset

test_dir = '/tmp/test_pytorch_hdf_parallel_io'


class TestHdf5IO(unittest.TestCase):

    """Test case docstring."""

    @classmethod
    def setUpClass(self):
        os.makedirs(test_dir)
        fp = os.path.join(test_dir, 'test{}.hdf5')
        for i in range(10):
            fn = fp.format(i)
            with h5py.File(fn, 'w') as f:
                for j in range(10):
                    name = "{}/{}"
                    array = np.arange(10, dtype=np.float32)+100*i+10*j
                    f.create_dataset(
                        name.format(j, 'value'),
                        data=array)
                    f.create_dataset(
                        name.format(j, 'key'),
                        data=100*i+10*j)
                f.close()

    @classmethod
    def tearDownClass(self):
        files = os.listdir(test_dir)
        for fp in files:
            os.remove(os.path.join(test_dir, fp))
        os.rmdir(test_dir)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ds_sep1(self):
        fp = os.path.join(test_dir, 'test{}.hdf5')
        for i in range(10):
            ds = HdfDataset(fp.format(i))
            dl = DataLoader(ds, batch_size=4, num_workers=1,
                            worker_init_fn=None, shuffle=False)
            for data in dl:
                key = data['key'].numpy()
                val = data['value'][:, 0].numpy()
                self.assertTrue(np.allclose(key, val))

    def test_ds_sep2(self):
        fp = os.path.join(test_dir, 'test{}.hdf5')
        for i in range(10):
            ds = HdfDataset(fp.format(i), num_workers=3)
            dl = DataLoader(ds, batch_size=4, num_workers=ds.num_workers,
                            worker_init_fn=ds._init_fn, shuffle=False)
            for data in dl:
                key = data['key'].numpy()
                val = data['value'][:, 0].numpy()
                self.assertTrue(np.allclose(key, val))

    def test_ds_concat(self):
        fp = os.path.join(test_dir, 'test{}.hdf5')
        DS = []
        for i in range(10):
            ds = HdfDataset(fp.format(i), num_workers=3)
            DS.append(ds)
        ds = HdfConcatDataset(DS)
        dl = DataLoader(ds, batch_size=4, num_workers=DS[0].num_workers,
                        worker_init_fn=ds._init_fn, shuffle=False)
        for data in dl:
            key = data['key'].numpy()
            val = data['value'][:, 0].numpy()
            self.assertTrue(np.allclose(key, val))


if __name__ == "__main__":
    unittest.main()
