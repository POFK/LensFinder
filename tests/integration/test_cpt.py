#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from lens.checkpoint import save, load

import unittest
from unittest import mock


class TestCpt(unittest.TestCase):

    """Checkpoint test"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch("torch.save")
    def test_save(self, torch_save):
        path = 'path_to_save'
        epoch = 1
        model = mock.MagicMock()
        loss = mock.MagicMock()
        device = torch.device('cpu')
        save_dict = {}
        save_dict['epoch'] = epoch
        save_dict['model_state_dict'] = model.state_dict()
        save_dict['loss'] = loss
        save_dict['device'] = device
        save(path, epoch, model, loss, device, opt=None)
    #   torch_save.assert_called_once_with(save_dict, path)

    @mock.patch("torch.load")
    def test_load(self, torch_load):
        torch_load = mock.MagicMock()
        path = 'path_to_save'
        model = mock.MagicMock()
        opt = mock.MagicMock()
        args = mock.MagicMock()
        args.use_cuda=False
        load(path, model, args=args, opt=None)

if __name__ == "__main__":
    unittest.main()
