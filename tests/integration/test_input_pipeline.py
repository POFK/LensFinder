#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import unittest
from unittest import mock

from lens.input_pipeline import MyDataset, RandomCrop, ToTensor, get_data


@mock.patch.object(np, "load")
class TestInputPipeline(unittest.TestCase):

    """Test case docstring."""

    @classmethod
    def setUpClass(self):
        self.data = {'image': np.random.randn(
            10, 101, 101, 3), 'label': np.arange(10)}
        self.data['image'][:, 0, 0, 0] = np.arange(10)

    @classmethod
    def tearDownClass(self):
        self.data = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_DataLoader(self, mock_load, crop_size=84):
        mock_load.return_value = self.data
        dataset = MyDataset(path='path_to_file')
        dl1, dl2 = get_data(dataset, dataset, 4, 3)
        for i in dl1:
            self.assertEqual(i['label'].shape[0], 4)
            image = i['image'].numpy()[:, 0, 0, 0]
            label = i['label'].numpy().reshape(-1)
            self.assertListEqual(list(image), list(label))
        for j in dl2:
            try:
                self.assertEqual(j['label'].shape[0], 3)
            except:
                self.assertEqual(j['label'].shape[0], 10 % 3)
            image = j['image'].numpy()[:, 0, 0, 0]
            label = j['label'].numpy().reshape(-1)
            self.assertListEqual(list(image), list(label))


if __name__ == "__main__":
    unittest.main()
