#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import torch
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--num_class', type=int, default=2, metavar='N')
parser.add_argument('--crop_range', type=int, default=84, metavar='N')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=128 * 4, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--epoch', type=int, default=0, metavar='N',
                    help='first epoch of training or the epoch used to evalation(default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--name', type=str, default='test', help='')
parser.add_argument('--mode', type=str, default='train', help='')
parser.add_argument('--model_dir', type=str, default='model',
                    help='For Saving the current Model')
parser.add_argument('--log_dir', type=str, default='log',
                    help='For Saving the current log information')
parser.add_argument('--base_dir',
                    type=str,
                    default='/data/storage1/LensFinder',
                    help='base Dir')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
#args.base_dir = os.path.join(args.base_dir, args.name)
args.use_cuda = use_cuda
