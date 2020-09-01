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

import pandas as pd
import time

import patsy
from combat import *
import anndata as ad

from sklearn.decomposition import PCA as sk_PCA


def do_batch_correction(df, embeds_cols, batch_corrections, verbose=False):
    df_corr = df.copy()
    end = time.time()
    if batch_corrections is not None:
        for corr in batch_corrections:
            if corr == "TVN":
                df_corr = correct_tvn(df_corr.copy(), embeds_cols, verbose)
            elif corr == "COMBAT":
                df_corr = correct_combat(df_corr.copy(), embeds_cols, verbose)
            else:
                print("Batch correction {} is not implemented".format(corr))

    if verbose:
        print('batch correction time: {0:.0f} s'.format(time.time() - end))

    return df_corr


# ------------------------------------------
# tvn
def correct_tvn(df, embeds_cols, verbose=False):
    if verbose:
        print('Do TVN')

    dmso = df.loc[(df['compound'] == 'DMSO'), embeds_cols].to_numpy(copy=True)
    p = sk_PCA(n_components=len(embeds_cols), whiten=True).fit(dmso)
    df.loc[:, embeds_cols] = p.transform(df.loc[:, embeds_cols])
    return df


# combat
def correct_combat(df, embeds_cols, verbose=False):
    if verbose:
        print('Do COMBAT')

    # Expression
    exp = df[embeds_cols].T

    # Covariants
    mod = patsy.dmatrix("~ compound + compound_uM", df, return_type="dataframe")
    ebat = combat(exp, df['plate'], mod, "compound_uM")

    df.loc[:, embeds_cols] = ebat.T
    return df
