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

import os
import pandas as pd
import numpy as np
import os
import math
import time

import itertools

from sklearn.decomposition import PCA as sk_PCA
from sklearn.manifold import TSNE as sk_TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import completeness_score, adjusted_rand_score, fowlkes_mallows_score, adjusted_mutual_info_score

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from hdbscan import HDBSCAN

np.random.seed(seed=42)

from correction import do_batch_correction
from plot import plot_confusion_matrix, plot_consistency_matrix, plot_clustermap, plot_cluster_assignment
from plot import plot_distance_heatmaps, plot_dmso_pca, plot_embeddings, plot_DMSO_3PCA


def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))


def number_of_components_95(df, embeds_cols):
    # PCA on all embeddings
    pca = sk_PCA().fit(df[embeds_cols])

    # Find the number of dimensions to explain 95% of variance
    i = 0
    s = 0
    for j in range(100):
        s += pca.explained_variance_ratio_[j]
        if s > 0.95:
            i = j
            break
    # There should be at least 8 dimensions        
    if i < 8:
        return 8
    else:
        return i


# --------------------------------------------------- collapse methods ---------------------------------------------------
def collapse_domain(df, headers, do_median=True, remove_dmso=False):
    if do_median:
        avg_df = df.groupby(headers).median()
    else:
        avg_df = df.groupby(headers).mean()
    avg_df = avg_df.reset_index(drop=False)
    if 'field' in avg_df.columns:
        avg_df = avg_df.drop(columns=['field', 'image_nr'])

    if remove_dmso:
        avg_df = avg_df[avg_df['compound'] != 'DMSO']
        avg_df = avg_df.reset_index(drop=True)
    return avg_df


def collapse_well_level(df, do_median=True, remove_dmso=False):
    avg_df = collapse_domain(df, ['batch', 'plate', 'well', 'compound', 'compound_uM', 'pseudoclass', 'moa'], do_median,
                             remove_dmso)
    return avg_df


def collapse_treatment_level(df, do_median=True, remove_dmso=False):
    avg_df = collapse_domain(df, ['compound', 'compound_uM', 'pseudoclass', 'moa'], do_median, remove_dmso)
    #avg_df = avg_df.drop(columns=['table_nr', 'replicate'])
    return avg_df


def collapse_plate_level(df, do_median=True, remove_dmso=False):
    avg_df = collapse_domain(df, ['batch', 'plate', 'compound', 'compound_uM', 'pseudoclass', 'moa'], do_median,
                             remove_dmso)
    return avg_df


def collapse_batch_level(df, do_median=True, remove_dmso=False):
    avg_df = collapse_domain(df, ['batch', 'compound', 'compound_uM', 'pseudoclass', 'moa'], do_median, remove_dmso)
    avg_df = avg_df.drop(columns=['replicate'])
    return avg_df


# --------------------------------------------------- eval methods ---------------------------------------------------
def calc_theoretical_number_of_clusters(df_well):
    n_cluster_dict = dict()
    # number of clusters according prior knowledge
    n_cluster_dict['n_cluster_comp'] = len(df_well['compound'].unique()) - 1
    n_cluster_dict['n_cluster_treat'] = len(df_well['pseudoclass'].unique())

    # number of clusters
    n_cluster_dict['n_cluster_treat'] = math.sqrt(len(df_well) / 2)
    return pd.DataFrame(n_cluster_dict)


def jaccard(labels1, labels2):
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)


def internal_partitional_validation(org_emb, tsne_emb, moa_, pred_, random_):
    metrices_dict = dict()
    # Calisnki-Harabasz coefficient
    metrices_dict['cal-har_moa'] = [calinski_harabasz_score(org_emb, moa_)]
    metrices_dict['cal-har_pred'] = [calinski_harabasz_score(org_emb, pred_)]
    metrices_dict['cal-har_rand'] = [calinski_harabasz_score(org_emb, random_)]

    # Calisnki-Harabasz coefficient on TSNE
    metrices_dict['cal-har_moa_tsne'] = [calinski_harabasz_score(tsne_emb, moa_)]
    metrices_dict['cal-har_pred_tsne'] = [calinski_harabasz_score(tsne_emb, pred_)]
    metrices_dict['cal-har_rand_tsne'] = [calinski_harabasz_score(tsne_emb, random_)]

    # Silhouette Coefficient (Cosine)
    metrices_dict['silhou_moa'] = [silhouette_score(org_emb, moa_)]
    metrices_dict['silhou_pred'] = [silhouette_score(org_emb, pred_)]
    metrices_dict['silhou_rand'] = [silhouette_score(org_emb, random_)]

    # Silhouette Coefficient on TSNE (Cosine)
    metrices_dict['silhou_moa_tsne'] = [silhouette_score(tsne_emb, moa_)]
    metrices_dict['silhou_pred_tsne'] = [silhouette_score(tsne_emb, pred_)]
    metrices_dict['silhou_rand_tsne'] = [silhouette_score(tsne_emb, random_)]
    return pd.DataFrame(metrices_dict)


def internal_hierarchical_validation(org_emb, tsne_emb):
    metrices_dict = dict()
    # Cophenetic distance
    X = pdist(org_emb, metric='cosine')
    Z = linkage(X, 'average')
    c, _ = cophenet(Z, X)
    metrices_dict['cophenet'] = [c]

    # Cophenetic distance on TSNE
    X = pdist(tsne_emb, metric='cosine')
    Z = linkage(X, 'average')
    c, _ = cophenet(Z, X)
    metrices_dict['cophenet_tsne'] = [c]
    return pd.DataFrame(metrices_dict)


def external_validation(pred_, moa_, treat_, comp_, random_, same_):
    metrices_dict = dict()
    # Completeness
    metrices_dict['comple_moa-pred'] = [completeness_score(moa_, pred_)]
    metrices_dict['comple_treat-moa'] = [completeness_score(treat_, moa_)]
    metrices_dict['comple_treat-pred'] = [completeness_score(treat_, pred_)]
    metrices_dict['comple_treat-rand'] = [completeness_score(treat_, random_)]
    metrices_dict['comple_treat-same'] = [completeness_score(treat_, same_)]
    metrices_dict['comple_comp-moa'] = [completeness_score(comp_, moa_)]
    metrices_dict['comple_comp_pred'] = [completeness_score(comp_, pred_)]
    metrices_dict['comple_comp_rand'] = [completeness_score(comp_, random_)]
    metrices_dict['comple_comp_same'] = [completeness_score(comp_, same_)]

    #  Jaccard similarity coefficient
    metrices_dict['jaccard_moa-pred'] = [jaccard(moa_, pred_)]
    metrices_dict['jaccard_treat-moa'] = [jaccard(treat_, moa_)]
    metrices_dict['jaccard_treat-pred'] = [jaccard(treat_, pred_)]
    metrices_dict['jaccard_treat-rand'] = [jaccard(treat_, random_)]
    metrices_dict['jaccard_treat-same'] = [jaccard(treat_, same_)]
    metrices_dict['jaccard_comp-moa'] = [jaccard(comp_, moa_)]
    metrices_dict['jaccard_comp_pred'] = [jaccard(comp_, pred_)]
    metrices_dict['jaccard_comp_rand'] = [jaccard(comp_, random_)]
    metrices_dict['jaccard_comp_same'] = [jaccard(comp_, same_)]

    # Adjusted Rand index
    metrices_dict['adj-rand_moa-pred'] = [adjusted_rand_score(moa_, pred_)]
    metrices_dict['adj-rand_treat-moa'] = [adjusted_rand_score(treat_, moa_)]
    metrices_dict['adj-rand_treat-pred'] = [adjusted_rand_score(treat_, pred_)]
    metrices_dict['adj-rand_treat-rand'] = [adjusted_rand_score(treat_, random_)]
    metrices_dict['adj-rand_treat-same'] = [adjusted_rand_score(treat_, same_)]
    metrices_dict['adj-rand_comp-moa'] = [adjusted_rand_score(comp_, moa_)]
    metrices_dict['adj-rand_comp_pred'] = [adjusted_rand_score(comp_, pred_)]
    metrices_dict['adj-rand_comp_rand'] = [adjusted_rand_score(comp_, random_)]
    metrices_dict['adj-rand_comp_same'] = [adjusted_rand_score(comp_, same_)]

    # Fowlkes-Mallows index
    metrices_dict['fow-mal_moa-pred'] = [fowlkes_mallows_score(moa_, pred_)]
    metrices_dict['fow-mal_treat-moa'] = [fowlkes_mallows_score(treat_, moa_)]
    metrices_dict['fow-mal_treat-pred'] = [fowlkes_mallows_score(treat_, pred_)]
    metrices_dict['fow-mal_treat-rand'] = [fowlkes_mallows_score(treat_, random_)]
    metrices_dict['fow-mal_treat-same'] = [fowlkes_mallows_score(treat_, same_)]
    metrices_dict['fow-mal_comp-moa'] = [fowlkes_mallows_score(comp_, moa_)]
    metrices_dict['fow-mal_comp_pred'] = [fowlkes_mallows_score(comp_, pred_)]
    metrices_dict['fow-mal_comp_rand'] = [fowlkes_mallows_score(comp_, random_)]
    metrices_dict['fow-mal_comp_same'] = [fowlkes_mallows_score(comp_, same_)]

    # Adjusted mutual information
    metrices_dict['adj-mut_moa-pred'] = [adjusted_mutual_info_score(moa_, pred_)]
    metrices_dict['adj-mut_treat-moa'] = [adjusted_mutual_info_score(treat_, moa_)]
    metrices_dict['adj-mut_treat-pred'] = [adjusted_mutual_info_score(treat_, pred_)]
    metrices_dict['adj-mut_treat-rand'] = [adjusted_mutual_info_score(treat_, random_)]
    metrices_dict['adj-mut_treat-same'] = [adjusted_mutual_info_score(treat_, same_)]
    metrices_dict['adj-mut_comp-moa'] = [adjusted_mutual_info_score(comp_, moa_)]
    metrices_dict['adj-mut_comp_pred'] = [adjusted_mutual_info_score(comp_, pred_)]
    metrices_dict['adj-mut_comp_rand'] = [adjusted_mutual_info_score(comp_, random_)]
    metrices_dict['adj-mut_comp_same'] = [adjusted_mutual_info_score(comp_, same_)]
    return pd.DataFrame(metrices_dict)


def batch_classification_accuracy(df_dmso, embeds_cols):
    clf = LogisticRegression(random_state=42, max_iter=1000)
    X = preprocessing.StandardScaler().fit_transform(df_dmso[embeds_cols])
    y_batch = df_dmso['batch']
    y_plate = df_dmso['plate']
    scores_batch = cross_val_score(clf, X, y_batch, cv=3)
    scores_plate = cross_val_score(clf, X, y_plate, cv=3)
    return pd.DataFrame([[scores_batch.mean(), scores_batch.std() * 2, scores_plate.mean(), scores_plate.std() * 2]],
                        columns=["batch_class_acc", "batch_class_std", "plate_class_acc", "plate_class_std"])


def NSC_k_NN(df_treatment, embeds_cols, plot_conf=False, savepath=None):
    # Create classes for each moa
    class_dict = dict(zip(df_treatment['moa'].unique(), np.arange(len(df_treatment['moa'].unique()))))
    df_treatment['moa_class'] = df_treatment['moa'].map(class_dict)

    # Create nearest neighbors classifier
    predictions = list()
    labels = list()
    label_names = list()
    for comp in df_treatment['compound'].unique():
        df_ = df_treatment.loc[df_treatment['compound'] != comp, :]
        knn = KNeighborsClassifier(n_neighbors=4, algorithm='brute', metric='cosine')
        knn.fit(df_.loc[:, embeds_cols], df_.loc[:, 'moa_class'])

        nn = knn.kneighbors(df_treatment.loc[df_treatment['compound'] == comp, embeds_cols])
        for p in range(nn[1].shape[0]):
            predictions.append(list(df_.iloc[nn[1][p]]['moa_class']))
        labels.extend(df_treatment.loc[df_treatment['compound'] == comp, 'moa_class'])
        label_names.extend(df_treatment.loc[df_treatment['compound'] == comp, 'moa'])

    predictions = np.asarray(predictions)
    k_nn_acc = [accuracy_score(labels, predictions[:, 0]),
                accuracy_score(labels, predictions[:, 1]),
                accuracy_score(labels, predictions[:, 2]),
                accuracy_score(labels, predictions[:, 3])]

    if plot_conf:
        print('There are {} treatments'.format(len(df_treatment)))
        print('NSC is: {:.2f}%'.format(accuracy_score(labels, predictions[:, 0]) * 100))
        plot_confusion_matrix(labels, predictions[:, 0], class_dict, 'NSC', savepath)
    return k_nn_acc


def NSB_k_NN(df_treatment, embeds_cols, plot_conf=False, savepath=None):
    # Remove moa with only 1 plate
    df_treatment = df_treatment[df_treatment['moa'] != 'Cholesterol-lowering']
    df_treatment = df_treatment[df_treatment['moa'] != 'Kinase inhibitors']
    df_treatment = df_treatment.reset_index(drop=True)

    class_dict = dict(zip(df_treatment['moa'].unique(), np.arange(len(df_treatment['moa'].unique()))))
    df_treatment['moa_class'] = df_treatment['moa'].map(class_dict)

    predictions = list()
    labels = list()
    label_names = list()
    for batch in df_treatment['table_nr'].unique():
        for comp in df_treatment.loc[df_treatment['table_nr'] == batch, 'compound'].unique():
            df_ = df_treatment.loc[(df_treatment['compound'] != comp) & (df_treatment['table_nr'] != batch), :]
            knn = KNeighborsClassifier(n_neighbors=4, algorithm='brute', metric='cosine')
            knn.fit(df_.loc[:, embeds_cols], df_.loc[:, 'moa_class'])

            nn = knn.kneighbors(
                df_treatment.loc[(df_treatment['compound'] == comp) & (df_treatment['table_nr'] == batch), embeds_cols])
            for p in range(nn[1].shape[0]):
                predictions.append(list(df_.iloc[nn[1][p]]['moa_class']))
            labels.extend(
                df_treatment.loc[(df_treatment['compound'] == comp) & (df_treatment['table_nr'] == batch), 'moa_class'])
            label_names.extend(
                df_treatment.loc[(df_treatment['compound'] == comp) & (df_treatment['table_nr'] == batch), 'moa'])

    predictions = np.asarray(predictions)
    k_nn_acc = [accuracy_score(labels, predictions[:, 0]),
                accuracy_score(labels, predictions[:, 1]),
                accuracy_score(labels, predictions[:, 2]),
                accuracy_score(labels, predictions[:, 3])]

    if plot_conf:
        print('There are {} treatments'.format(len(df_treatment)))
        print('NSCB is: {:.2f}%'.format(accuracy_score(labels, predictions[:, 0]) * 100))
        plot_confusion_matrix(labels, predictions[:, 0], class_dict, 'NSCB', savepath)
    return k_nn_acc


def NSC(df_well, df_plate, df_batch, embeds_cols):
    nsc_well = NSC_k_NN(df_well, embeds_cols)
    nsc_plate = NSC_k_NN(df_plate, embeds_cols)
    nsc_batch = NSC_k_NN(df_batch, embeds_cols)
    nsc_average = np.asarray([nsc_well, nsc_plate, nsc_batch]).mean(axis=0)

    nsc_list = list()
    nsc_list.extend(nsc_well)
    nsc_list.extend(nsc_plate)
    nsc_list.extend(nsc_batch)
    nsc_list.extend(nsc_average)

    return pd.DataFrame([nsc_list], columns=['NSC_1-NN_well', 'NSC_2-NN_well', 'NSC_3-NN_well', 'NSC_4-NN_well',
                                             'NSC_1-NN_plate', 'NSC_2-NN_plate', 'NSC_3-NN_plate', 'NSC_4-NN_plate',
                                             'NSC_1-NN_batch', 'NSC_2-NN_batch', 'NSC_3-NN_batch', 'NSC_4-NN_batch',
                                             'NSC_1-NN_avg', 'NSC_2-NN_avg', 'NSC_3-NN_avg', 'NSC_4-NN_avg'])


def NSB(df_well, df_plate, df_batch, embeds_cols):
    nsb_well = NSB_k_NN(df_well, embeds_cols)
    nsb_plate = NSB_k_NN(df_plate, embeds_cols)
    nsb_batch = NSB_k_NN(df_batch, embeds_cols)
    nsb_average = np.asarray([nsb_well, nsb_plate, nsb_batch]).mean(axis=0)

    nsb_list = list()
    nsb_list.extend(nsb_well)
    nsb_list.extend(nsb_plate)
    nsb_list.extend(nsb_batch)
    nsb_list.extend(nsb_average)

    return pd.DataFrame([nsb_list], columns=['NSB_1-NN_well', 'NSB_2-NN_well', 'NSB_3-NN_well', 'NSB_4-NN_well',
                                             'NSB_1-NN_plate', 'NSB_2-NN_plate', 'NSB_3-NN_plate', 'NSB_4-NN_plate',
                                             'NSB_1-NN_batch', 'NSB_2-NN_batch', 'NSB_3-NN_batch', 'NSB_4-NN_batch',
                                             'NSB_1-NN_avg', 'NSB_2-NN_avg', 'NSB_3-NN_avg', 'NSB_4-NN_avg'])


def create_consistency_matrix(df_well, predictions, savepath):
    # create mappers
    pred_mapper = {"cluster_{}".format(i): i for i in sorted(list(predictions))}
    moa_mapper = dict(zip(sorted(df_well['moa'].unique()), range(len(df_well['moa'].unique()))))
    treatment_mapper = dict(zip(sorted(df_well['pseudoclass'].unique()), range(len(df_well['pseudoclass'].unique()))))
    compound_mapper = dict(zip(sorted(df_well['compound'].unique()), range(len(df_well['compound'].unique()))))

    pred_ = list(predictions)
    moa_ = list(df_well['moa'].map(moa_mapper))
    treat_ = list(df_well['pseudoclass'].map(treatment_mapper))
    comp_ = list(df_well['compound'].map(compound_mapper))

    # prediction vs moa
    conti_pred_moa = metrics.cluster.contingency_matrix(pred_, moa_)
    df_conti_pred_moa = pd.DataFrame(conti_pred_moa,
                                     columns=list(moa_mapper.keys()),
                                     index=list(pred_mapper.keys()))
    plot_consistency_matrix(df_conti_pred_moa, "prediction-moa", savepath)

    # moa vs treatment
    conti_moa_treat = metrics.cluster.contingency_matrix(moa_, treat_)
    df_conti_moa_treat = pd.DataFrame(conti_moa_treat,
                                      columns=list(treatment_mapper.keys()),
                                      index=list(moa_mapper.keys()))
    plot_consistency_matrix(df_conti_moa_treat, "moa-treatment", savepath)

    # prediction vs treatment
    conti_pred_treat = metrics.cluster.contingency_matrix(pred_, treat_)
    df_conti_pred_treat = pd.DataFrame(conti_pred_treat,
                                       columns=list(treatment_mapper.keys()),
                                       index=list(pred_mapper.keys()))
    plot_consistency_matrix(df_conti_pred_treat, "prediction-treatment", savepath)

    # prediction vs compound
    conti_pred_comp = metrics.cluster.contingency_matrix(pred_, comp_)
    df_conti_pred_comp = pd.DataFrame(conti_pred_comp,
                                       columns=list(compound_mapper.keys()),
                                       index=list(pred_mapper.keys()))
    plot_consistency_matrix(df_conti_pred_comp, "prediction-compound", savepath)


# --------------------------------------------------- clustering assignment ---------------------------------------------------
def assign_clusters(df_well, embeds_cols, min_cluster_size=10, min_samples=3):
    pca_image = sk_PCA(n_components=number_of_components_95(df_well, embeds_cols)).fit_transform(df_well[embeds_cols])
    tsne_image = sk_TSNE(metric='cosine', n_jobs=1).fit_transform(pca_image)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='manhattan', min_samples=min_samples).fit(tsne_image)

    # tsne_image = sk_TSNE(metric='cosine', n_jobs=1).fit_transform(df_well[embeds_cols])
    # clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='manhattan', min_samples=min_samples).fit(tsne_image)
    return clusterer.labels_, clusterer.labels_.max(), tsne_image


# --------------------------------------------------- MAIN EVALUATION ---------------------------------------------------
def evaluate_epoch(df_tile, embeds_cols, verbose=False):
    if verbose:
        print('Start evaluating of the features')

    end = time.time()

    # ----------- Well level -----------
    # Create well collapse dataframe
    df_well = collapse_well_level(df_tile.copy(), remove_dmso=True)

    # clustering
    predictions, n_clusters, pca_tsne_image = assign_clusters(df_well, embeds_cols, min_cluster_size=10, min_samples=3)
    n_clus_df = pd.DataFrame([n_clusters], columns=['n_clusters_well'])

    # create mappers
    moa_mapper = dict(zip(sorted(df_well['moa'].unique()), range(len(df_well['moa'].unique()))))
    treatment_mapper = dict(zip(sorted(df_well['pseudoclass'].unique()), range(len(df_well['pseudoclass'].unique()))))
    compound_mapper = dict(zip(sorted(df_well['compound'].unique()), range(len(df_well['compound'].unique()))))

    # create assignment lists
    pred_ = list(predictions)
    moa_ = list(df_well['moa'].map(moa_mapper))
    treat_ = list(df_well['pseudoclass'].map(treatment_mapper))
    comp_ = list(df_well['compound'].map(compound_mapper))
    random_ = list(np.random.randint(12, size=len(moa_)))
    same_ = list(np.ones(len(moa_)))

    if verbose:
        print('Run validation methods')

    # validation
    int_par_df = internal_partitional_validation(df_well[embeds_cols], pca_tsne_image, moa_, pred_, random_)
    int_hier_df = internal_hierarchical_validation(df_well[embeds_cols], pca_tsne_image)
    ext_df = external_validation(pred_, moa_, treat_, comp_, random_, same_)

    # Remove undefined clusters
    df_labeled = df_tile[df_tile['moa'] != 'undefined'].copy()
    df_labeled = df_labeled.reset_index(drop=True)

    # Create batch and plate collapse dataframe
    df_well = collapse_well_level(df_labeled.copy(), remove_dmso=True)
    df_plate = collapse_plate_level(df_labeled.copy(), remove_dmso=True)
    df_batch = collapse_batch_level(df_labeled.copy(), remove_dmso=True)

    # Nearest Neighborhood
    NSC_df = NSC(df_well, df_plate, df_batch, embeds_cols)

    NSB_df = NSB(df_well, df_plate, df_batch, embeds_cols)

    # ----------- Treatment level -----------
    # Average per treatment per plate and median per treatment per batch
    avg_df = collapse_plate_level(df_labeled.copy(), do_median=False)
    df_treatment = collapse_treatment_level(avg_df, do_median=True, remove_dmso=True)

    NSC_treatment_df = pd.DataFrame([NSC_k_NN(df_treatment, embeds_cols)],
                                    columns=['NSC_1-NN_treatment', 'NSC_2-NN_treatment', 'NSC_3-NN_treatment',
                                             'NSC_4-NN_treatment'])
    NSCB_treatment_df = pd.DataFrame([NSB_k_NN(df_treatment, embeds_cols)],
                                    columns=['NSB_1-NN_treatment', 'NSB_2-NN_treatment', 'NSB_3-NN_treatment',
                                             'NSB_4-NN_treatment'])

    # Create well DMSO dataframe
    df_dmso = df_tile.loc[(df_tile['compound'] == 'DMSO'), :].copy()
    df_dmso = df_dmso.reset_index(drop=True)

    # Batch effect
    batch_acc_df = batch_classification_accuracy(df_dmso, embeds_cols)

    if verbose:
        print('Evaluation time: {0:.2f} s'.format(time.time() - end))
    #  Return DataFrame with all metrices
    return pd.concat([n_clus_df, int_par_df, int_hier_df, ext_df, NSC_df, NSB_df, NSC_treatment_df, NSCB_treatment_df, batch_acc_df], axis=1)


def evaluate_training(df_tile, embeds_cols, savepath=None, verbose=False):
    if verbose:
        print('Start evaluating of best features')

    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    # ----------- Well level -----------
    # Create well collapse dataframe
    df_well_with_dmso = collapse_well_level(df_tile.copy(), remove_dmso=False)
    df_save_well = df_well_with_dmso.copy()

    # Plot embeddings with ground truth labels and assigned labeles
    moa_unique_list = sorted(unique(list(df_well_with_dmso['moa'])))
    pca_well, pca_tsne_well, tsne_well, umap_well = plot_embeddings(df_well_with_dmso, embeds_cols, moa_unique_list,
                                                                    savepath)

    # Save well values
    df_save_well['PCA1'] = pca_well[:, 0]
    df_save_well['PCA2'] = pca_well[:, 1]
    df_save_well['TSNE1'] = tsne_well[:, 0]
    df_save_well['TSNE2'] = tsne_well[:, 1]
    df_save_well['PCA_TSNE1'] = pca_tsne_well[:, 0]
    df_save_well['PCA_TSNE2'] = pca_tsne_well[:, 1]
    df_save_well['UMAP1'] = umap_well[:, 0]
    df_save_well['UMAP2'] = umap_well[:, 1]

    # Create well DMSO dataframe
    df_tile_dmso = df_tile.loc[(df_tile['compound'] == 'DMSO'), :].copy()
    df_tile_dmso = df_tile_dmso.reset_index(drop=True)
    df_well_dmso = df_well_with_dmso.loc[(df_well_with_dmso['compound'] == 'DMSO'), :].copy()
    df_well_dmso = df_well_dmso.reset_index(drop=True)

    # Plot DMSO embeddings
    batch_unique_list = sorted(unique(list(df_well_with_dmso['batch'])))
    plot_dmso_pca(df_tile_dmso, df_well_dmso, embeds_cols, batch_unique_list, savepath)
    plot_distance_heatmaps(df_tile_dmso, df_well_dmso, embeds_cols, savepath)
    plot_DMSO_3PCA(df_tile_dmso, embeds_cols, savepath)

    # clustering wells
    df_well = collapse_well_level(df_tile.copy(), remove_dmso=True)
    predictions, n_clusters, pca_tsne_image = assign_clusters(df_well, embeds_cols, min_cluster_size=10, min_samples=3)
    plot_cluster_assignment(pca_tsne_image, predictions, list(df_well['moa']), savepath, prefix="Well_")

    # Save clustering assignment
    df_well['cluster_nr'] = predictions
    df_well['PCA_TSNE1'] = pca_tsne_image[:, 0]
    df_well['PCA_TSNE2'] = pca_tsne_image[:, 1]

    # Plot consistency_matrix
    create_consistency_matrix(df_well, predictions, savepath)

    # ----------- Treatment level -----------
    # Average per treatment per plate and median per treatment per batch
    avg_df = collapse_plate_level(df_tile.copy(), do_median=False)
    df_treatment = collapse_treatment_level(avg_df, do_median=True, remove_dmso=True)

    # clustering treatments
    predictions2, n_clusters2, pca_tsne_image2 = assign_clusters(df_treatment, embeds_cols, min_cluster_size=5, min_samples=3)
    plot_cluster_assignment(pca_tsne_image2, predictions2, list(df_treatment['moa']), savepath, prefix="Treatment_")

    # Save clustering assignment
    df_treatment['cluster_nr'] = predictions2
    df_treatment['PCA_TSNE1'] = pca_tsne_image2[:, 0]
    df_treatment['PCA_TSNE2'] = pca_tsne_image2[:, 1]

    # Labeled evaluation
    df_treatment = df_treatment[df_treatment['moa'] != 'undefined'].copy()
    df_treatment = df_treatment.reset_index(drop=True)
    plot_clustermap(df_treatment, embeds_cols, savepath)

    # NSC and NSCB
    NSC_k_NN(df_treatment, embeds_cols, plot_conf=True, savepath=savepath)
    NSB_k_NN(df_treatment, embeds_cols, plot_conf=True, savepath=savepath)
    return df_save_well
