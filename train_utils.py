#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from glob import glob
from logging import DEBUG
from logging import FileHandler
from logging import Formatter
from logging import getLogger
from logging import StreamHandler
from logging import WARNING
import os
import time
import torch
import yaml
from torch.autograd import Variable

logger = getLogger('training')


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("Takes {} seconds.".format(elapse))
    return _measure_time


def load_config(config_path):
    """Load a configration yaml file.

    Args:
        config_path (str):
    Returns:
        params (dict):

    """
    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    params = conf['param']
    return params


def save_config(conf, save_path):
    """Save a configuration file as a yaml file.

    Args:
        conf (dict):

    """
    with open(os.path.join(save_path), "w") as f:
        f.write(yaml.dump({'param': conf}, default_flow_style=False))


def set_logger(save_path, key, stdout=False):
    """Set logger.

    Args:
        save_path (str):
        key (str):
        stdout (bool):
    Returns:
        logger ():

    """
    logger = getLogger(key)
    sh = StreamHandler()
    fh = FileHandler(save_path)

    logger.setLevel(DEBUG)
    sh.setLevel(WARNING)
    fh.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s %(name)s line:%(lineno)d %(levelname)s: %(message)s')
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    if stdout:
        logger.info = lambda x: print(x)

    return logger


def set_save_path(save_path):
    """Change directory name to avoid name ovarlapping.

    Args:
        save_path (str):
    Returns:
        save_path_new (str):

    """
    # Reset model directory
    model_idx = 0
    save_path_new = save_path
    while True:
        if os.path.isfile(os.path.join(save_path_new, 'conf.yml')):
            # Training of the first model have not been finished yet
            model_idx += 1
            save_path_new = save_path + '_' + str(model_idx)
        else:
            break
    if not os.path.isdir(save_path_new):
        os.mkdir(save_path_new)
    return save_path_new


def load_checkpoint(model, checkpoint_path, optimizer=None, resume=False):
    """Load checkpoint.

    Args:
        model (torch.nn.Module):
        checkpoint_path (str): path to the saved model (model..epoch-*)
        epoch (int): negative values mean the offset from the last saved model
        optimizer ():
        resume (bool): if True, restore the save optimizer
    Returns:
        model (torch.nn.Module):
        optimizer ():

    """
    if not os.path.isfile(checkpoint_path):
        raise ValueError('There is no checkpoint')

    epoch = int(os.path.basename(checkpoint_path).split('-')[-1]) - 1

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("No checkpoint found at %s" % checkpoint_path)

    # Restore parameters
    logger.info("=> Loading checkpoint (epoch:%d): %s" % (epoch + 1, checkpoint_path))
    model.load_state_dict(checkpoint['state_dict'])

    # Restore optimizer
    if resume:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(0)
                        # state[k] = v.cuda(self.device_id)
                        # TODO(hirofumi): Fix for multi-GPU
            # NOTE: from https://github.com/pytorch/pytorch/issues/2830
        else:
            raise ValueError('Set optimizer.')

    return model, optimizer


def save_checkpoint(model, save_path, optimizer, epoch, remove_old_checkpoints=False):
    """Save checkpoint.

    Args:
        model (torch.nn.Module):
        save_path (str): path to the directory to save a model
        optimizer (LRScheduler): optimizer
        epoch (int): currnet epoch
        remove_old_checkpoints (bool): if True, all checkpoints
            other than the best one will be deleted

    """
    cpu = 1
    model_path = os.path.join(save_path, 'model.epoch-' + str(epoch))

    # Remove old checkpoints
    if remove_old_checkpoints:
        for path in glob(os.path.join(save_path, 'model.epoch-*')):
            os.remove(path)

    # Save parameters, optimizer, step index etc.
    if cpu == 1:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),  # LR scheduler
        }
    else:
        checkpoint = {
            "state_dict": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),  # LR scheduler
        }
    torch.save(checkpoint, model_path)

    logger.info("=> Saved checkpoint (epoch:%d): %s" % (epoch, model_path))
