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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import pdist, squareform

np.random.seed(seed=42)

def unique(list1):
    x = np.array(list1)
    return list(np.unique(x))


def number_of_components_95(df, embeds_cols):
    # PCA on all embeddings
    pca = PCA().fit(df[embeds_cols])

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


def plot_dmso_pca(df_tile_dmso, df_well_dmso, embeds_cols, batch_unique_list, savepath):
    pca_tile_dmso = PCA(2, whiten=True).fit_transform(df_tile_dmso.loc[:, embeds_cols])
    pca_well_dmso = PCA(2, whiten=True).fit_transform(df_well_dmso.loc[:, embeds_cols])

    fig, axs = plt.subplots(ncols=2, figsize=(24, 12))
    sns.scatterplot(
        pca_tile_dmso[:, 0],
        pca_tile_dmso[:, 1],
        hue=list(df_tile_dmso['batch']),
        hue_order=batch_unique_list,
        s=70,
        ax=axs[0]
    ).set_title("PCA DMSO - tile level")
    sns.scatterplot(
        pca_well_dmso[:, 0],
        pca_well_dmso[:, 1],
        hue=list(df_well_dmso['batch']),
        hue_order=batch_unique_list,
        s=70,
        ax=axs[1]
    ).set_title("PCA DMSO - well level")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "dmso_pca.png"))
    plt.show()


def plot_embeddings(df_well, embeds_cols, moa_unique_list, savepath):
    pca_well = PCA(n_components=2, whiten=True).fit_transform(df_well[embeds_cols])
    pca95_well = PCA(n_components=number_of_components_95(df_well, embeds_cols)).fit_transform(df_well[embeds_cols])
    pca_tsne_well = TSNE(metric='cosine', n_jobs=1).fit_transform(pca95_well)
    tsne_well = TSNE(metric='cosine', n_jobs=1).fit_transform(df_well[embeds_cols])
    umap_well = UMAP(metric='cosine').fit_transform(df_well[embeds_cols])

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(24, 24))
    sns.scatterplot(
        pca_well[:, 0],
        pca_well[:, 1],
        hue=list(df_well['moa']),
        hue_order=moa_unique_list,
        s=70,
        ax=axs[0, 0]
    ).set_title("PCA - well level")
    sns.scatterplot(
        pca_tsne_well[:, 0],
        pca_tsne_well[:, 1],
        hue=list(df_well['moa']),
        hue_order=moa_unique_list,
        s=70,
        ax=axs[0, 1]
    ).set_title("PCA95% + TSNE - well level")
    sns.scatterplot(
        tsne_well[:, 0],
        tsne_well[:, 1],
        hue=list(df_well['moa']),
        hue_order=moa_unique_list,
        s=70,
        ax=axs[1, 0]
    ).set_title("TSNE - well level")
    sns.scatterplot(
        umap_well[:, 0],
        umap_well[:, 1],
        hue=list(df_well['moa']),
        hue_order=moa_unique_list,
        s=70,
        ax=axs[1, 1]
    ).set_title("UMAP - well level")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "embeddings.png"))
    plt.show()
    return pca_well, pca_tsne_well, tsne_well, umap_well


def plot_distance_heatmaps(df_tile_dmso, df_well_dmso, embeds_cols, savepath):
    dis_metric_tile = squareform(pdist(df_tile_dmso.loc[:, embeds_cols].to_numpy(copy=True), metric='cosine'))
    dis_metric_well = squareform(pdist(df_well_dmso.loc[:, embeds_cols].to_numpy(copy=True), metric='cosine'))

    fig, axs = plt.subplots(ncols=2, figsize=(24, 12))
    sns.heatmap(
        ax=axs[0],
        data=dis_metric_tile,
        vmin=0,
        cmap=sns.color_palette("Blues"),
        square=True,
        xticklabels=False,
        yticklabels=False
    ).set_title("cosine distance of dmso data - tile level")
    sns.heatmap(
        ax=axs[1],
        data=dis_metric_well,
        vmin=0,
        cmap=sns.color_palette("Blues"),
        square=True,
        xticklabels=False,
        yticklabels=False
    ).set_title("cosine distance of dmso data - well level")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "dmso_distance_heatmaps.png"))
    plt.show()


def plot_clustermap(df_treatment, embeds_cols, savepath):
    # Cosine distance heatmap
    dis_metric = squareform(pdist(df_treatment.loc[:, embeds_cols].to_numpy(copy=True), metric='cosine'))

    # Plot
    lut = dict(zip(sorted(df_treatment['moa'].unique()),
                   sns.hls_palette(len(df_treatment['moa'].unique()), h=0.6, l=0.5, s=0.8)))
    legend_TN = [mpatches.Patch(color=c, label=l) for l, c in lut.items()]

    g = sns.clustermap(
        data=dis_metric,
        # metric='cosine',
        row_colors=list(df_treatment['moa'].map(lut)),
        col_colors=list(df_treatment['moa'].map(lut)),
        cmap='coolwarm',
        figsize=(18, 18),
        linewidths=0.01,
        xticklabels=False,
        yticklabels=False
    )  # .set_title("Hierarchical Clustering - treatment level - {}".format(title))
    l2 = g.ax_heatmap.legend(loc='center left', bbox_to_anchor=(1.01, 0.85), handles=legend_TN, frameon=True)
    g.savefig(os.path.join(savepath, "hierarchical_clustering.png"))
    plt.show()


def plot_confusion_matrix(labels, predictions, class_dict, title, savepath):
    # Create Confusion matrix
    conf = confusion_matrix(labels, predictions)
    conf_df = pd.DataFrame(conf, columns=list(class_dict.keys()), index=list(class_dict.keys()))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        ax=ax,
        data=conf_df,
        cmap='bwr',
        linewidths=0.01,
        linecolor='black',
        center=0,
        square=True,
        annot=True,
        cbar=False
    ).set_title("Confusion metrics - {}".format(title))

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "confusion_matrix_{}.svg".format(title)))
    plt.show()


def plot_cluster_assignment(tsne, prediction, gt_moa, savepath, prefix=""):
    print('HDBScan found ', prediction.max(), 'clusters')
    fig, axs = plt.subplots(ncols=3, figsize=(45, 15))
    sns.scatterplot(
        tsne[:, 0],
        tsne[:, 1],
        hue=gt_moa,
        s=70,
        ax=axs[0]
    ).set_title("PCA 95% + TSNE - Groundtruth MoA")
    sns.scatterplot(
        tsne[:, 0],
        tsne[:, 1],
        hue=["cluster_{}".format(i) for i in list(prediction)],
        s=70,
        ax=axs[1],
        legend=False
    ).set_title("PCA 95% + TSNE - HDBSCAN")
    sns.scatterplot(
        tsne[:, 0],
        tsne[:, 1],
        hue=list(prediction == -1),
        s=70,
        ax=axs[2]
    ).set_title("PCA 95% + TSNE - Outliers")
    plt.tight_layout()
    #fig.savefig(os.path.join(savepath, prefix + "cluster_assignment.svg"))
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        tsne[:, 0],
        tsne[:, 1],
        hue=gt_moa,
        hue_order=list(sorted(unique(gt_moa))),
        #s=70,
        ax=ax
    )#.set_title("PCA 95% + TSNE - Groundtruth MoA")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, prefix + "cluster_assignment.svg"))
    plt.show()


# Plot confusion matrix
def plot_consistency_matrix(df_conti, title, savepath):
    c, r = df_conti.shape
    size = (10 + int(r/2), 10 + int(c/2))
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        ax=ax,
        data=df_conti,
        cmap='bwr',
        linewidths=0.01,
        linecolor='black',
        center=0,
        square=True,
        annot=True,
        cbar=False
    ).set_title("contingency metrics {}".format(title))

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "consistency_matrix_{}.png".format(title)))
    plt.show()


def plot_DMSO_3PCA(df_tile_dmso, embeds_cols, savepath):
    pca_tile_dmso = PCA(whiten=True).fit_transform(df_tile_dmso.loc[:, embeds_cols])

    df_d = pd.concat([df_tile_dmso["table_nr"], pd.DataFrame(pca_tile_dmso[:, :3], columns=["PCA1", "PCA2", "PCA3"])],
                     axis=1)
    df_d = df_d.rename(columns={"table_nr": "Batch"})
    fig, axs = plt.subplots(nrows=3, figsize=(8, 15))
    sns.boxplot(x="Batch", y="PCA1", data=df_d, ax=axs[0]).set(ylim=(-4, 4))
    sns.boxplot(x="Batch", y="PCA2", data=df_d, ax=axs[1]).set(ylim=(-4, 4))
    sns.boxplot(x="Batch", y="PCA3", data=df_d, ax=axs[2]).set(ylim=(-4, 4))
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "DMSO_3PCA.png"))
    plt.show()


def plot_training(df, savepath):
    fig, axs = plt.subplots(ncols=2, figsize=(24, 12))
    sns.lineplot(x='epoch',
                 y="loss",
                 data=df,
                 ax=axs[0]
                 ).set_title("Training loss")
    sns.lineplot(x='epoch',
                 y="nmi",
                 data=df,
                 ax=axs[1]
                 ).set_title("NMI t/t-1")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "training.png"))
    plt.show()


def plot_training_summary(df, savepath):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(24, 24))
    sns.lineplot(x='epoch',
                 y="NSC_1-NN_treatment",
                 data=df,
                 ax=axs[0, 0]
                 ).set_title("1st NSC (treatment)")
    sns.lineplot(x='epoch',
                 y="n_clusters",
                 data=df,
                 ax=axs[0, 1]
                 ).set_title("Number of predicted clusters")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['comple_treat-moa', 'comple_treat-pred', 'comple_treat-rand',
                                          'comple_treat-same'],
                              var_name='metric'),
                 ax=axs[0, 2]
                 ).set_title("Completeness (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'], value_vars=['silhou_moa', 'silhou_pred', 'silhou_rand'],
                              var_name='metric'),
                 ax=axs[1, 0]
                 ).set_title("Silhouette Coefficient")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-rand_treat-moa', 'adj-rand_treat-pred', 'adj-rand_treat-rand',
                                          'adj-rand_treat-same'],
                              var_name='metric'),
                 ax=axs[1, 1]
                 ).set_title("Adjusted Rand index (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-mut_treat-moa', 'adj-mut_treat-pred', 'adj-mut_treat-rand',
                                          'adj-mut_treat-same'],
                              var_name='metric'),
                 ax=axs[1, 2]
                 ).set_title("Adjusted mutual information (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_avg', 'NSC_2-NN_avg', 'NSC_3-NN_avg', 'NSC_4-NN_avg'],
                              var_name='metric'),
                 ax=axs[2, 0]
                 ).set_title("NSC")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_avg', 'NSB_2-NN_avg', 'NSB_3-NN_avg', 'NSB_4-NN_avg'],
                              var_name='metric'),
                 ax=axs[2, 1]
                 ).set_title("NSB")
    sns.lineplot(x='epoch',
                 y="batch_class_acc",
                 data=df,
                 ax=axs[2, 2]
                 ).set_title("Batch classification accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "training_summary.png"))
    plt.show()


def plot_inter_part_validation(df, savepath):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24, 24))
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'], value_vars=['cal-har_moa', 'cal-har_pred', 'cal-har_rand'],
                              var_name='metric'),
                 ax=axs[0, 0]
                 ).set_title("Calisnki-Harabasz coefficient")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['cal-har_moa_tsne', 'cal-har_pred_tsne', 'cal-har_rand_tsne'],
                              var_name='metric'),
                 ax=axs[0, 1]
                 ).set_title("Calisnki-Harabasz coefficient on TSNE")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'], value_vars=['silhou_moa', 'silhou_pred', 'silhou_rand'],
                              var_name='metric'),
                 ax=axs[1, 0]
                 ).set_title("Silhouette Coefficient")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['silhou_moa_tsne', 'silhou_pred_tsne', 'silhou_rand_tsne'],
                              var_name='metric'),
                 ax=axs[1, 1]
                 ).set_title("Silhouette Coefficient on TSNE")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "inter_part_validation.png"))
    plt.show()


def plot_total_score(df, savepath):
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(24, 48))
    sns.lineplot(x='epoch',
                 y="n_clusters",
                 data=df,
                 ax=axs[0, 0]
                 ).set_title("Number of clusters")
    sns.lineplot(x='epoch',
                 y="adj-mut_moa-pred",
                 data=df,
                 ax=axs[0, 1]
                 ).set_title("Adjusted Mutual Information Pred vs MoA")
    sns.lineplot(x='epoch',
                 y="comple_treat-pred",
                 data=df,
                 ax=axs[1, 0]
                 ).set_title("Completeness treatment vs prediction")
    sns.lineplot(x='epoch',
                 y="adj-rand_moa-pred",
                 data=df,
                 ax=axs[1, 1]
                 ).set_title("Adjusted Rand Index Pred vs MoA")
    sns.lineplot(x='epoch',
                 y="silhou_pred_tsne",
                 data=df,
                 ax=axs[2, 0]
                 ).set_title("Silhouette Coefficient")
    sns.lineplot(x='epoch',
                 y='jaccard_moa-pred',
                 data=df,
                 ax=axs[2, 1]
                 ).set_title("Jaccard similarity coefficient Pred vs MoA")
    sns.lineplot(x='epoch',
                 y="NSC_1-NN_treatment",
                 data=df,
                 ax=axs[3, 0]
                 ).set_title("NSC treatment")
    sns.lineplot(x='epoch',
                 y="total_score",
                 data=df,
                 ax=axs[3, 1]
                 ).set_title("Total score")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "total_score.png"))
    plt.show()


def plot_inter_hier_validation(df, savepath):
    fig, axs = plt.subplots(ncols=2, figsize=(24, 12))
    sns.lineplot(x='epoch',
                 y="cophenet",
                 data=df,
                 ax=axs[0]
                 ).set_title("Cophenetic distance")
    sns.lineplot(x='epoch',
                 y="cophenet_tsne",
                 data=df,
                 ax=axs[1]
                 ).set_title("Cophenetic distance on TSNE")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "inter_hier_validation.png"))
    plt.show()


def plot_external_validation(df, savepath):
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(24, 60))
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['comple_treat-moa', 'comple_treat-pred', 'comple_treat-rand',
                                          'comple_treat-same'],
                              var_name='metric'),
                 ax=axs[0, 0]
                 ).set_title("Completeness (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['comple_comp-moa', 'comple_comp_pred', 'comple_comp_rand',
                                          'comple_comp_same'],
                              var_name='metric'),
                 ax=axs[0, 1]
                 ).set_title("Completeness (compound)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['jaccard_treat-moa', 'jaccard_treat-pred', 'jaccard_treat-rand',
                                          'jaccard_treat-same'],
                              var_name='metric'),
                 ax=axs[1, 0]
                 ).set_title("Jaccard similarity coefficient (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['jaccard_comp-moa', 'jaccard_comp_pred', 'jaccard_comp_rand',
                                          'jaccard_comp_same'],
                              var_name='metric'),
                 ax=axs[1, 1]
                 ).set_title("Jaccard similarity coefficient (compound)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-rand_treat-moa', 'adj-rand_treat-pred', 'adj-rand_treat-rand',
                                          'adj-rand_treat-same'],
                              var_name='metric'),
                 ax=axs[2, 0]
                 ).set_title("Adjusted Rand index (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-rand_comp-moa', 'adj-rand_comp_pred', 'adj-rand_comp_rand',
                                          'adj-rand_comp_same'],
                              var_name='metric'),
                 ax=axs[2, 1]
                 ).set_title("Adjusted Rand index (compound)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['fow-mal_treat-moa', 'fow-mal_treat-pred', 'fow-mal_treat-rand',
                                          'fow-mal_treat-same'],
                              var_name='metric'),
                 ax=axs[3, 0]
                 ).set_title("Fowlkes-Mallows index (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['fow-mal_comp-moa', 'fow-mal_comp_pred', 'fow-mal_comp_rand',
                                          'fow-mal_comp_same'],
                              var_name='metric'),
                 ax=axs[3, 1]
                 ).set_title("Fowlkes-Mallows index (compound)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-mut_treat-moa', 'adj-mut_treat-pred', 'adj-mut_treat-rand',
                                          'adj-mut_treat-same'],
                              var_name='metric'),
                 ax=axs[4, 0]
                 ).set_title("Adjusted mutual information (treatment)")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['adj-mut_comp-moa', 'adj-mut_comp_pred', 'adj-mut_comp_rand',
                                          'adj-mut_comp_same'],
                              var_name='metric'),
                 ax=axs[4, 1]
                 ).set_title("Adjusted mutual information (compound)")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "external_validation.png"))
    plt.show()


def plot_NSC_NSB(df, savepath):
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(24, 72))
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_well', 'NSC_2-NN_well', 'NSC_3-NN_well', 'NSC_4-NN_well'],
                              var_name='metric'),
                 ax=axs[0, 0]
                 ).set_title("NSC well")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_well', 'NSB_2-NN_well', 'NSB_3-NN_well', 'NSB_4-NN_well'],
                              var_name='metric'),
                 ax=axs[0, 1]
                 ).set_title("NSB well")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_plate', 'NSC_2-NN_plate', 'NSC_3-NN_plate', 'NSC_4-NN_plate'],
                              var_name='metric'),
                 ax=axs[1, 0]
                 ).set_title("NSC plate")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_plate', 'NSB_2-NN_plate', 'NSB_3-NN_plate', 'NSB_4-NN_plate'],
                              var_name='metric'),
                 ax=axs[1, 1]
                 ).set_title("NSB plate")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_batch', 'NSC_2-NN_batch', 'NSC_3-NN_batch', 'NSC_4-NN_batch'],
                              var_name='metric'),
                 ax=axs[2, 0]
                 ).set_title("NSC batch")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_batch', 'NSB_2-NN_batch', 'NSB_3-NN_batch', 'NSB_4-NN_batch'],
                              var_name='metric'),
                 ax=axs[2, 1]
                 ).set_title("NSB batch")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_avg', 'NSC_2-NN_avg', 'NSC_3-NN_avg', 'NSC_4-NN_avg'],
                              var_name='metric'),
                 ax=axs[3, 0]
                 ).set_title("NSC average")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_avg', 'NSB_2-NN_avg', 'NSB_3-NN_avg', 'NSB_4-NN_avg'],
                              var_name='metric'),
                 ax=axs[3, 1]
                 ).set_title("NSB average")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSC_1-NN_treatment', 'NSC_2-NN_treatment', 'NSC_3-NN_treatment', 'NSC_4-NN_treatment'],
                              var_name='metric'),
                 ax=axs[4, 0]
                 ).set_title("NSC treatment")
    sns.lineplot(x='epoch',
                 y="value",
                 hue='metric',
                 data=pd.melt(df, id_vars=['epoch'],
                              value_vars=['NSB_1-NN_treatment', 'NSB_2-NN_treatment', 'NSB_3-NN_treatment', 'NSB_4-NN_treatment'],
                              var_name='metric'),
                 ax=axs[4, 1]
                 ).set_title("NSB treatment")
    sns.lineplot(x='epoch',
                 y="batch_class_acc",
                 data=df,
                 ax=axs[5, 0]
                 ).set_title("Batch classification accuracy")
    sns.lineplot(x='epoch',
                 y="plate_class_acc",
                 data=df,
                 ax=axs[5, 1]
                 ).set_title("Plate classification accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, "NSC_NSB.png"))
    plt.show()
