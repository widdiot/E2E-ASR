#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frame stacking."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from asr import Dataset
import numpy as np
from torch_utils import np2tensor
from torch_utils import pad_list


def stack_frame(feat, n_stacks, n_skips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        feat (list): `[T, input_dim]`
        n_stacks (int): the number of frames to stack
        n_skips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / n_skips), input_dim * n_stacks]`

    """
    if n_stacks == 1 and n_stacks == 1:
        return feat

    if n_stacks < n_skips:
        raise ValueError('n_skips must be less than n_stacks.')

    n_frames, input_dim = feat.shape
    n_frames_new = (n_frames + 1) // n_skips

    stacked_feat = np.zeros((n_frames_new, input_dim * n_stacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(feat):
        if t == len(feat) - 1:  # final frame
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(n_frames_new):
                # Concatenate stacked frames
                for i in range(len(stack)):
                    stacked_feat[stack_count][input_dim
                                              * i:input_dim * (i + 1)] = stack[i]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(n_skips):
                    if len(stack) != 0:
                        stack.pop(0)

        elif len(stack) < n_stacks:  # first & middle frames
            # Stack some frames until stack is filled
            stack.append(frame_t)

        if len(stack) == n_stacks:
            # Concatenate stacked frames
            for i in range(n_stacks):
                stacked_feat[stack_count][input_dim
                                          * i:input_dim * (i + 1)] = stack[i]
            stack_count += 1

            # Delete some frames to skip
            for _ in range(n_skips):
                stack.pop(0)

    return stacked_feat


def stack_frame_T(feat, n_stacks, n_skips, dtype=np.float32):
    """Stack & skip some frames. This implementation is based on

       https://arxiv.org/abs/1507.06947.
           Sak, Haşim, et al.
           "Fast and accurate recurrent neural network acoustic models for speech recognition."
           arXiv preprint arXiv:1507.06947 (2015).

    Args:
        feat (list): `[T, input_dim]`
        n_stacks (int): the number of frames to stack
        n_skips (int): the number of frames to skip
        dtype ():
    Returns:
        stacked_feat (np.ndarray): `[floor(T / n_skips), input_dim * n_stacks]`

    """
    if n_stacks == 1 and n_stacks == 1:
        return feat

    if n_stacks < n_skips:
        raise ValueError('n_skips must be less than n_stacks.')

    n_frames, input_dim = feat.shape
    n_frames_new = (n_frames + 1) // n_skips

    stacked_feat = np.zeros((n_frames_new, input_dim * n_stacks), dtype=dtype)
    stack_count = 0
    stack = []
    for t, frame_t in enumerate(feat):
        if t == len(feat) - 1:  # final frame
            # Stack the final frame
            stack.append(frame_t)

            while stack_count != int(n_frames_new):
                # Concatenate stacked frames
                for i in range(len(stack)):
                    stacked_feat[stack_count][input_dim
                                              * i:input_dim * (i + 1)] = stack[i]
                stack_count += 1

                # Delete some frames to skip
                for _ in range(n_skips):
                    if len(stack) != 0:
                        stack.pop(0)

        elif len(stack) < n_stacks:  # first & middle frames
            # Stack some frames until stack is filled
            stack.append(frame_t)

        if len(stack) == n_stacks:
            # Concatenate stacked frames
            for i in range(n_stacks):
                stacked_feat[stack_count][input_dim
                                          * i:input_dim * (i + 1)] = stack[i]
            stack_count += 1

            # Delete some frames to skip
            for _ in range(n_skips):
                stack.pop(0)

    return stacked_feat

if __name__ == '__main__':
    train_set = Dataset(corpus='hindi',
                        tsv_path="/home/asir/kaldi/egs/mini_librispeech/s5/data/dataset/train_clean_5_5_wpbpe30000.tsv",
                        dict_path="/home/asir/kaldi/egs/mini_librispeech/s5/data/dict/train_clean_5_wpbpe30000.txt",
                        unit='wp',
                        wp_model="/home/asir/kaldi/egs/mini_librispeech/s5/data/dict/train_clean_5_bpe30000.model",
                        batch_size=50,  # * args.n_gpus,
                        n_epochs=25,
                        min_n_frames=40,
                        max_n_frames=2000,
                        sort_by='input',
                        short2long=True,
                        sort_stop_epoch=100,
                        dynamic_batching=True,
                        subsample_factor=1,
                        discourse_aware=False,
                        skip_thought=False)
    batch, is_new_epoch = train_set.next()
    xs, ys, xlens = batch['xs'], batch['ys'], batch['xlens']
    xs = [stack_frame(x, 3, 3) for x in xs]
    xs = [np2tensor(x).float() for x in xs]
    xs = pad_list(xs, 0.0)
    print(xs.shape)
