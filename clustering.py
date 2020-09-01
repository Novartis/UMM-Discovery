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

import time
import numpy as np
import pandas as pd
import faiss

from scipy.sparse import csr_matrix, find
import skimage.external.tifffile as sktiff
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import my_transform

from sklearn.decomposition import PCA as sk_PCA
from umap import UMAP
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE as sk_TSNE
from hdbscan import HDBSCAN


__all__ = ['PIC', 'Kmeans', 'AdaptiveKmeans', 'HDBscan', 'cluster_assign', 'arrange_clustering']

class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = sktiff.imread(path).astype('float32')

        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

def preprocess_features(npdata, n_components=16, method='PCA', n_jobs=1):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    if method == 'PCA':
        mat = faiss.PCAMatrix(ndim, n_components, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    # Apply UMAP for dimensionality reduction
    elif method == 'UMAP':
        fit = UMAP(n_components=n_components, metric='cosine')
        npdata = np.ascontiguousarray(fit.fit_transform(npdata))
    # Apply T-SNE for dimensionality reduction
    elif method == 'TSNE':
        if n_components > 3:
            X = sk_PCA().fit_transform(npdata)
            PCAinit = X[:, :n_components] / np.std(X[:, 0]) * 0.0001
            fit = TSNE(n_components=n_components, init=PCAinit, n_jobs=n_jobs)
            npdata = np.ascontiguousarray(fit.fit_transform(npdata), dtype='float32')
        else:
            fit = sk_TSNE(n_components=n_components, metric='cosine', n_jobs=n_jobs)
            npdata = np.ascontiguousarray(fit.fit_transform(npdata))
    # Apply adaptive T-SNE for dimensionality reduction
    elif method == 'AdaptiveTSNE':
            pca = sk_PCA().fit(npdata)

            # Find all the eigenvectors that explain 95% of the variance
            i = 0
            s = 0
            for j in range(len(pca.explained_variance_ratio_)):
                s += pca.explained_variance_ratio_[j]
                if s > 0.95:
                    i = j
                    # Prevent smaller than 8
                    if i < 8:
                        i = 8
                    break

            # Fit and transform the data with the number of components that explain 95%
            pca95_well = sk_PCA(n_components=i).fit_transform(npdata)

            # Do a similarity measure with TSNE on the pca data
            if n_components > 3:
                PCAinit = pca95_well[:, :n_components] / np.std(pca95_well[:, 0]) * 0.0001
                fit = TSNE(n_components=n_components, init=PCAinit, n_jobs=n_jobs)
                npdata = np.ascontiguousarray(fit.fit_transform(pca95_well))
            else:
                fit = sk_TSNE(n_components=n_components, metric='cosine', n_jobs=n_jobs)
                npdata = np.ascontiguousarray(fit.fit_transform(pca95_well))

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    return npdata

def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D

def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    tra = [my_transform.random_horizontal_flip(),
           my_transform.random_vertical_flip(),
           my_transform.random_180_rotation(),
           transforms.ToTensor()]
           
    t = transforms.Compose(tra)

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    return [int(n[0]) for n in I], 0


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans:
    def __init__(self, k, dim_method="PCA", n_components=16, n_jobs=1):
        self.k = k
        self.dim_method = dim_method
        self.n_components = n_components
        self.n_jobs = n_jobs

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        data = np.ascontiguousarray(data)
        xb = preprocess_features(data, n_components=self.n_components, method=self.dim_method, n_jobs=self.n_jobs)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


class AdaptiveKmeans:
    def __init__(self, initial_k, end_k, end_epochs, initial_epoch=0, dim_method="PCA", n_components=16, n_jobs=1):
        self.initial_k = initial_k
        self.end_k = end_k
        self.end_epochs = end_epochs
        self.dim_method = dim_method
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.decay_rate = -(initial_k - end_k) / end_epochs
        self.epoch = initial_epoch

    def cluster_decay(self):
        if self.epoch < self.end_epochs:
            n_clusters = self.decay_rate * self.epoch + self.initial_k
        else:
            n_clusters = self.end_k
        return int(n_clusters)

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        data = np.ascontiguousarray(data)
        xb = preprocess_features(data, n_components=self.n_components, method=self.dim_method, n_jobs=self.n_jobs)

        # Number of clusters
        k = self.cluster_decay()

        # cluster the data
        I, loss = run_kmeans(xb, k, verbose)
        self.images_lists = [[] for i in range(k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        self.epoch += 1

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwith of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if (i == 200 - 1):
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC():
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwith of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True, dim_method="PCA", n_components=16, n_jobs=1):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons
        self.dim_method = dim_method
        self.n_components = n_components
        self.n_jobs = n_jobs

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        data = np.ascontiguousarray(data)
        xb = preprocess_features(data, n_components=self.n_components, method=self.dim_method, n_jobs=self.n_jobs)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0


def run_hdbscan(x, min_cluster_size, min_samples):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # perform dbscan
    dbsc = HDBSCAN(min_cluster_size = min_cluster_size, metric='manhattan', min_samples = min_samples).fit(x)

    # perform the training
    labels = dbsc.labels_
    max_clust = max(labels)

    return labels, max_clust


class HDBscan:
    """Class to perform HDBscan clustering.
        Args:
            min_cluster_size:
            nnn:
            distribute_outliers:
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """
    def __init__(self, args=None, min_cluster_size=15, min_samples=5, nnn=30, distribute_outliers=True, dim_method="PCA", n_components=16, n_jobs=1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.nnn = nnn
        self.distribute_outliers = distribute_outliers
        self.dim_method = dim_method
        self.n_components = n_components
        self.n_jobs = n_jobs

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        print("preprocess features")
        data = np.ascontiguousarray(data)
        xb = preprocess_features(data, n_components=self.n_components, method=self.dim_method, n_jobs=self.n_jobs)

        # construct nnn graph
        print("Construct graph")
        I, _ = make_graph(xb, self.nnn)

        # run dbscan
        print("Run HDBscan")
        clust, max_clust = run_hdbscan(xb, self.min_cluster_size, self.min_samples)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        print("Distribute outliers")
        if self.distribute_outliers:
            # check if there are outliers
            if -1 in clust:
                while(len(images_lists[-1])>0):
                    n_outliers = len(images_lists[-1])
                    print(n_outliers)
                    clust_NN = {}
                    # loop over all outliers (are all in the last cluster)
                    for s in images_lists[-1]:
                        # for NN
                        for n in I[s, 1:]:
                            # if NN is not a singleton
                            if not n in images_lists[-1]:
                                clust_NN[s] = n
                                break
                    for s in clust_NN:
                        images_lists[-1].remove(s)
                        clust[s] = clust[clust_NN[s]]
                        images_lists[clust[s]].append(s)

                    # make sure it is not stuck in a loop
                    if len(images_lists[-1]) == n_outliers:
                        max_clust += 1
                        d = images_lists[-1][0]
                        images_lists[max_clust] = [d]
                        images_lists[-1].remove(d)

                del images_lists[-1]

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('Number of clusters: {}'.format(len(np.unique(clust))))
            print('hdbscan time: {0:.0f} s'.format(time.time() - end))
        return 0
