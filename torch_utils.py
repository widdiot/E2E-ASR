#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def tensor2np(x):
    """Convert torch.Tensor to np.ndarray.

    Args:
        x (Tensor):
    Returns:
        np.ndarray

    """
    return x.cpu().numpy()


def np2tensor(array, device_id=-1):
    """Convert form np.ndarray to torch.Tensor.

    Args:
        array (np.ndarray): A tensor of any sizes
        device_id (int): ht index of the device
    Returns:
        tensor (FloatTensor/IntTensor/LongTensor):

    """
    tensor = torch.from_numpy(array)
    if device_id >= 0:
        tensor = tensor.cuda(device_id)
    return tensor


def pad_list(xs, pad_value=0.0, pad_left=False):
    """Convert list of Tensors to a single Tensor with padding.

    Args:
        xs (list): A list of length `[B]`, which concains Tensors of size `[T, input_size]`
        pad_value (float):
        pad_left (bool):
    Returns:
        xs_pad (FloatTensor): `[B, T, input_size]`

    """
    bs = len(xs)
    max_time = max(x.size(0) for x in xs)
    xs_pad = xs[0].new_zeros(bs, max_time, * xs[0].size()[1:]).fill_(pad_value)
    for b in range(bs):
        if len(xs[b]) == 0:
            continue
        if pad_left:
            xs_pad[b, -xs[b].size(0):] = xs[b]
        else:
            xs_pad[b, :xs[b].size(0)] = xs[b]
    return xs_pad


def make_pad_mask(seq_lens, device_id=-1):
    """Make mask for padding.

    Args:
        seq_lens (IntTensor): `[B]`
        device_id (int):
    Returns:
        mask (IntTensor): `[B, T]`

    """
    bs = seq_lens.size(0)
    max_time = max(seq_lens)

    seq_range = torch.arange(0, max_time, dtype=torch.int32)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
    seq_length_expand = seq_range_expand.new(seq_lens).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand

    if device_id >= 0:
        mask = mask.cuda(device_id)

    return mask


def compute_accuracy(logits, ys_ref, pad):
    """Compute accuracy.

    Args:
        logits (FloatTensor): `[B, T, vocab]`
        ys_ref (LongTensor): `[B, T]`
        pad (int): index for padding
    Returns:
        acc (float): teacher-forcing accuracy

    """
    pad_pred = logits.view(ys_ref.size(0), ys_ref.size(1), logits.size(-1)).argmax(2)
    mask = ys_ref != pad
    numerator = torch.sum(pad_pred.masked_select(mask) == ys_ref.masked_select(mask))
    denominator = torch.sum(mask)
    acc = float(numerator) * 100 / float(denominator)
    return acc
