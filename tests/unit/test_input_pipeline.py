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

    def test_init(self, mock_load):
        mock_load.return_value = self.data
        dataset = MyDataset(path='path_to_file')
        mock_load.assert_called_once_with('path_to_file')

    def test_trans_tensor(self, mock_load):
        mock_load.return_value = self.data
        preprocess = transforms.Compose([ToTensor(10)])
        dataset = MyDataset(path='path_to_file', transform=preprocess)
        for i in range(10):
            self.assertTrue(torch.is_tensor(dataset[i]['label']))
            self.assertTrue(torch.is_tensor(dataset[i]['image']))

    def test_trans_crop(self, mock_load, crop_size=84):
        mock_load.return_value = self.data
        preprocess = transforms.Compose([RandomCrop(crop_size)])
        dataset = MyDataset(path='path_to_file', transform=preprocess)
        for i in range(10):
            self.assertTrue(list(dataset[i]['image'].shape) == [
                            crop_size, crop_size, 3])

    def test_trans_prep(self, mock_load, crop_size=84):
        mock_load.return_value = self.data
        preprocess = transforms.Compose([RandomCrop(crop_size), ToTensor(10)])
        dataset = MyDataset(path='path_to_file', transform=preprocess)
        for i in range(10):
            self.assertTrue(torch.is_tensor(dataset[i]['label']))
            self.assertTrue(torch.is_tensor(dataset[i]['image']))
            self.assertTrue(list(dataset[i]['image'].shape) == [
                            3, crop_size, crop_size])

if __name__ == "__main__":
    unittest.main()
