# UMM Discovery


## Overview


[UMM-Discovery](https://www.biorxiv.org/content/10.1101/2020.07.22.215459v1.full) is a fully unsupervised deep learning method to cluster cellular images with similar phenotypes together, solely based on the intensity values. It is a modification of the Deep Clustering framework developed by *Caron at al. (2019)*. Based on the findings of *Godinez et al. (2017)*, we decided to use an updated version of the Deep Neural Network (DNN) architecture, called Multi-Scale-Net. UMM Discovery uses two batch correction methods, Typical Variation Normalization (TVN) (Ando et al., 2017) and Combat (Johnson et al., 2007), during training to significantly improve the results and to create more representative embeddings.


### Link to resources

UMM Discovery makes use of:

* [DeepCluster](https://github.com/facebookresearch/deepcluster), by Facebook, Inc, available under a [Creative Commons Attribution-Noncommercial](https://creativecommons.org/licenses/by-nc/4.0/) license.
* [ComBat](https://github.com/brentp/combat.py) by brent
* [Multi-Scale-Net](https://academic.oup.com/bioinformatics/article/33/13/2010/2997285) from Godinez et al. (2017)

## Prequisites and dependencies

* All Requirements of [DeepCluster](https://github.com/facebookresearch/deepcluster)
* [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
* [HDBScan](https://github.com/scikit-learn-contrib/hdbscan)
* patsy
* anndata
* sklearn
* seaborn
* umap
* matplotlib

## Installation

The easiest way to install all dependencies is with conda.

`$ conda env create -f environment.yml`

* [ComBat](https://github.com/brentp/combat.py)

Clone the github for [ComBat](https://github.com/brentp/combat.py) from brentb and copy the combat.py in the directory

## Data

The method can be applied on any cellular dataset. To do so change the loading of the images in the my_dataset.py. Additionally, the Multi-Scale Net input shape may need some changes (see file model.py) if the number of input channels differ.
In the paper, UMM Discovery is evaluated on the [BBBC021](https://bbbc.broadinstitute.org/BBBC021) cellular dataset available from the Broad Bioimage Benchmark Collection.

## Running UMM Discovery

Start a jupyter session on your local machine or gpu cluster

 `$ jupyter`

and open jupyter notebook `UMM_discovery_BBBC021.ipynb`

Within the notebook change the parameters (e.g. dataset path and output path) to your needs and run the cells.

## Reference

If you use this code, please cite the following paper:

Rens Janssens, Xian Zhang, Audrey Kauffmann, Antoine de Weck, Eric Y. Durand. "Fully unsupervised deep mode of action learning for phenotyping high-content cellular images" doi: https://doi.org/10.1101/2020.07.22.215459

@article {Janssens2020.07.22.215459,
	author = {Janssens, Rens and Zhang, Xian and Kauffmann, Audrey and de Weck, Antoine and Durand, Eric Y.},
	title = {Fully unsupervised deep mode of action learning for phenotyping high-content cellular images},
	elocation-id = {2020.07.22.215459},
	year = {2020},
	doi = {10.1101/2020.07.22.215459},
	URL = {https://www.biorxiv.org/content/early/2020/07/23/2020.07.22.215459},
	eprint = {https://www.biorxiv.org/content/early/2020/07/23/2020.07.22.215459.full.pdf},
	journal = {bioRxiv}
}