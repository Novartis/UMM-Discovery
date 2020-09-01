# Copyright 2020 Novartis Institutes for BioMedical Research Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file incorporates work covered by the following copyright and  
# permission notice:
#
#   Copyright (c) 2017-present, Facebook, Inc.
#   All rights reserved.
#
#   This source code is licensed as Creative Commons Attribution-Noncommercial 
#   and can be found under https://creativecommons.org/licenses/by-nc/4.0/.

import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from model import MultiScaleNet

def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        num_features = checkpoint['n_input_dim']
        num_features = checkpoint['n_features']
        num_classes = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        model = MultiScaleNet(input_dim, num_features, out=int(num_classes[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """
    def __init__(self, path):
        self.path = path
        self.data = {
            'epoch' :[],
            'features' : [],
            'eval_metrices' : [],
            'cluster_assignments' : [],
            'loss' : [],
            'nmi' : [],
            'clustering' : []
        }

    def load_data(self, resume = None):
        with open(os.path.join(self.path), 'rb') as fp:
            self.data = pickle.load(fp)
        
        if resume:
            if resume < len(self.data['epoch']):
                self.data['epoch'] = self.data['epoch'][:resume]
                self.data['features'] = self.data['features'][:resume]
                self.data['eval_metrices'] = self.data['eval_metrices'][:resume]
                self.data['cluster_assignments'] = self.data['cluster_assignments'][:resume]
                self.data['loss'] = self.data['loss'][:resume]
                self.data['nmi'] = self.data['nmi'][:resume]
                self.data['clustering'] = self.data['clustering'][:resume]

    def log(self, epoch, features, eval_metrices, cluster_assignments, loss, nmi, clustering):
        self.data['epoch'].append(epoch)
        self.data['features'].append(features)
        self.data['eval_metrices'].append(eval_metrices)
        self.data['cluster_assignments'].append(cluster_assignments)
        self.data['loss'].append(loss)
        self.data['nmi'].append(nmi)
        self.data['clustering'].append(clustering)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

    def get_data(self):
        return self.data