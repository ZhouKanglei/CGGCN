#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/5/8 下午10:24

import sys

import torch

sys.path.extend(['../'])

import pickle
import numpy as np
from torch.utils.data import Dataset

from feeders import tools

class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 bone=False, vel=False, multi_input=False):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the beginning or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.bone = bone
        self.vel = vel
        self.multi_input = multi_input

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        if len(self.data.shape) == 3:
            N, T, _ = self.data.shape
            self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # modality transformation
        data_numpy_1 = data_numpy.copy()
        if self.bone:
            from .bone_pairs import kinetics_pairs
            for v1, v2 in kinetics_pairs:
                data_numpy_1[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

        if self.vel:
            data_numpy_1[:, :-1] = data_numpy_1[:, 1:] - data_numpy_1[:, :-1]
            data_numpy_1[:, -1] = 0

        # modality 2
        data_numpy_2 = data_numpy_1.copy()
        if not self.vel:
            data_numpy_2[:, :-1] = data_numpy_1[:, 1:] - data_numpy_1[:, :-1]
            data_numpy_2[:, -1] = 0
        elif self.bone:
            data_numpy_2 = data_numpy.copy()
        else:
            from .bone_pairs import kinetics_pairs
            for v1, v2 in kinetics_pairs:
                data_numpy_2[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

        # concatenation
        if self.multi_input:
            data_numpy_1 = torch.from_numpy(data_numpy_1).unsqueeze(0)
            data_numpy_2 = torch.from_numpy(data_numpy_2).unsqueeze(0)
            data_numpy_ = torch.cat([data_numpy_1, data_numpy_2], dim=0)
        else:
            data_numpy_ = torch.from_numpy(data_numpy_1)

        return data_numpy_, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)