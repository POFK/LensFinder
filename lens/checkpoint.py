#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn
from lens.flags import args


def save(path, epoch, model, loss, device, opt=None):
    save_dict = {}
    save_dict['epoch'] = epoch
    try:
        model.device_ids
        save_dict['model_state_dict'] = model.module.state_dict()
    except AttributeError:
        save_dict['model_state_dict'] = model.state_dict()
    save_dict['loss'] = loss
    save_dict['device'] = device
    if opt is not None:
        save_dict['optimizer_state_dict'] = opt.state_dict()
    torch.save(save_dict, path)


def load(path, model, args=args, opt=None):
    device = torch.device("cuda:0" if args.use_cuda else "cpu")
    map_location = "cpu" if device.type == "cpu" else "cuda:0"
    try:
        checkpoint = torch.load(path)
    except RuntimeError:
        checkpoint = torch.load(path, map_location=map_location)
        device_cpt = checkpoint['device']
        print("{} --> {}...".format(device_cpt, map_location))
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    device_cpt = checkpoint['device']
    if opt is not None:
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    try:
        model.device_ids
        model.module.load_state_dict(checkpoint['model_state_dict'])
    except AttributeError:
        model.load_state_dict(checkpoint['model_state_dict'])
    if device.type == "cuda":
        model.to(device)
    return model, epoch
