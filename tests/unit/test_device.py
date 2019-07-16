#!/usr/bin/env python
# coding=utf-8

import torch
import unittest
from unittest import mock
from lens.utils import get_device, get_model_device


def is_cuda_version():
    try:
        return torch.cuda.is_available()
    except AssertionError:
        return False


class TestDevice(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch('torch.cuda.device_count')
    @mock.patch('torch.cuda.is_available')
    def test_cpu(self, has_cuda, cuda_count):
        has_cuda.return_value = False
        cuda_count.return_value = 0
        device = get_device()
        self.assertSequenceEqual(device.type, 'cpu')
        self.assertIsNone(device.index)

    @mock.patch('torch.cuda.device_count')
    @mock.patch('torch.cuda.is_available')
    def test_gpu0(self, has_cuda, cuda_count):
        has_cuda.return_value = True
        cuda_count.return_value = 1
        device = get_device()
        self.assertSequenceEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)

    @mock.patch('torch.cuda.device_count')
    @mock.patch('torch.cuda.is_available')
    def test_gpu4(self, has_cuda, cuda_count):
        has_cuda.return_value = True
        cuda_count.return_value = 4
        device = get_device()
        self.assertSequenceEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)

    @unittest.skipUnless(is_cuda_version(), reason='pytorch is not gpu version')
    @unittest.skipUnless(torch.cuda.device_count()>1, reason='one GPU only')
    @mock.patch('torch.cuda.device_count')
    @mock.patch('torch.cuda.is_available')
    def test_gpu4_model(self, has_cuda, cuda_count):
        """test it later"""
        has_cuda.return_value = True
        cuda_count.return_value = 4
        device = get_device()
        model = mock.MagicMock()
        model_dev = get_model_device(model, device)
        self.assertListEqual(model_dev.device_ids,[0,1,2,3])

    @mock.patch('torch.cuda.device_count')
    @mock.patch('torch.cuda.is_available')
    def test_gpu4_notuse(self, has_cuda, cuda_count):
        has_cuda.return_value = True
        cuda_count.return_value = 4
        device = get_device(use_cuda=False)
        self.assertSequenceEqual(device.type, 'cpu')
        self.assertIsNone(device.index)


if __name__ == "__main__":
    unittest.main()
