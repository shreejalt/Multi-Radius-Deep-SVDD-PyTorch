# Appendix A: PyTorch Implementation of Multi Radius Deep SVDD
This code is built upon the original implementation of [Lukas Ruff](https://github.com/lukasruff/Deep-SVDD-PyTorch). We have added FINCH algorithm and Multi-radius code to incorporate the multi-radius learning of the clusters. Thanks to **Lukas Ruff** for the excellent codebase. The original paper was presented in ICML 2018 with the name of 'Deep One-Class Classification'.  


## Abstract
> > Our work is the complete extension of the work by  **Ruff.et al** Deep SVDD: Deep One Class Classification}. We observed some trends in different scenarios where this paper can be utilized, and all our observations are derived from that. Apart from our observation and changes, all the courtesy is to the authors . We try to extend the work done by the authors to multi-radius anomaly deep one class classification where we can cluster the embedding space to generate the pseudo-labels of the clusters and then use those class information for further learning the radii for each cluster instead of unique radius for all the classes. This method helps use to get more compact radii for each class and during the inference, we make the decision on the data to be an anomaly, if it is an outlier for all the classes. We have shown the proof of concept on the two well-known datasets of CIFAR10 and MNIST. Due to GPU and time constraints, we have mentioned anomaly detection results only for several classes on both the datasets. We developed our code on the original PyTorch implementation of the authors: [Link](https://github.com/lukasruff/Deep-SVDD-PyTorch)

## Installation

This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`. We have updated this file to support our codebase.

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Deep-SVDD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Deep-SVDD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running experiments

We currently have implemented the MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)) and 
CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) datasets and 
simple LeNet-type networks.

Have a look into `main.py` for all possible arguments and options. We have added `--use_multi_radius` flag which takes the cluster in account and train using the new DOC objective function.

We use different implementation of **FINCH** **algorithm** by [eren-ck](https://github.com/eren-ck/finch). Original python implementation can be found at [here](https://github.com/ssarfraz/FINCH-Clustering)

### MNIST example
```
cd <path-to-Deep-SVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3 --normal_class 0 --normal_class 1 --use_multi_radius True;
```
This example trains a One-Class Deep SVDD model where digit 1, 0, and 3  are considered to be the normal classes. Autoencoder pretraining is used for parameter initialization.

### CIFAR-10 example
```
cd <path-to-Deep-SVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/cifar10_test

# change to source directory
cd src

# run experiment
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3 --normal_class 1 --normal_class 0 --sue_multi_radius True;
```
This example trains a One-Class Deep SVDD model where cats 0, 1, and 3 are considered to be the normal classes. 
Autoencoder pretraining is used for parameter initialization.

## Citations

You find a PDF of the Deep One-Class Classification ICML 2018 paper at 
[http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html).

```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4393--4402},
  year      = {2018},
  volume    = {80},
}
```

```
@inproceedings{finch,
    author    = {M. Saquib Sarfraz and Vivek Sharma and Rainer Stiefelhagen}, 
    title     = {Efficient Parameter-free Clustering Using First Neighbor Relations}, 
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {8934--8943}
    year  = {2019}
}
```

