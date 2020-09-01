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

from torch.utils.data import Dataset
import skimage.external.tifffile as sktiff
import pandas as pd
import os
import numpy as np


class bbbc021_dataset(Dataset):
    def __init__(self, root_dir, metadata, label_header, transform=None):
        super(bbbc021_dataset, self).__init__()
        self.df = pd.read_csv(os.path.join(root_dir, metadata), sep='\t')
        self.root_dir = root_dir
        self.transform = transform
        self.label_header = label_header

        self.add_columns()

        self.class_to_idx = dict()
        self.classes = []
        i = 0
        for c in self.df[label_header].unique():
            self.classes.append(c)
            self.class_to_idx[c] = i
            i += 1

        self.imgs = []
        for row in range(len(self.df)):
            img_name = os.path.join(self.root_dir, self.df['pseudoclass'].iloc[row], self.df['filename'].iloc[row])
            label = self.class_to_idx[self.df[label_header].iloc[row]]
            item = (img_name, label)
            self.imgs.append(item)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        image = sktiff.imread(img_name).astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.df)

    def add_columns(self):
        rel_dose_list = list()
        for i, row in self.df.iterrows():
            compound = self.df.loc[i, 'compound']
            if (compound == 'DMSO'):
                rel_dose_list.append(0)
            else:
                d = self.df.loc[i, 'compound_uM'] / max(self.df.loc[self.df['compound'] == compound, 'compound_uM'])
                rel_dose_list.append(d)
        self.df['rel_dose_adjust'] = rel_dose_list
        self.df['compound'] = self.df['compound'].str.replace(',', '.')
        self.df['batch'] = 'Batch_' + self.df['table_nr'].map(str)

    def get_df(self):
        return self.df
