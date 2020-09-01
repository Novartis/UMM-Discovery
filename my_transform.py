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

import torch
import numpy as np
from random import sample, random


class z_score(object):
    def __call__(self, img):
        img = img.type(torch.FloatTensor)
        img = img - img.mean()
        img = img / img.std()
        return img

    def __repr__(self):
        return self.__class__.__name__ 


class random_180_rotation(object):
    def __call__(self, img):
        angle = sample([0, 2], 1)[0]
        return np.rot90(img, k=angle).copy()
        
    def __repr__(self):
        return self.__class__.__name__


class random_vertical_flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            return np.flip(img, axis=0)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class random_horizontal_flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            return np.flip(img, axis=1)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)