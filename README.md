# README #

# Welcome

This package is written by Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])). It contains methods of variational Latent Gaussian Process (vLGP) model based on Yuan Zhao and Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
It has been developed and implemented with the goal of recovering dynamics from population spike trains. 

The code needs further testing and subject to change.

## Usage

To get started, see the examples in notebook: ./tutorial.ipynb

## Modules

| module     | function                                                                                      |
|:-----------|-----------------------------------------------------------------------------------------------|
| core       | model fitting, prediction and validation                                                      |
| hyper      | hyperparameter optimization                                                                   |
| math       | link functions, incomplelte Cholesky decompostion, angle between subspaces, orthogonalization |
| plot       | raster and dynamics                                                                           |
| simulation | simulation of Gaussian process, Lorenz dynamics and spike trains                              |
| util       | lag matrix construction, rotation, load and save                                              |


## Contact

If you notice a bug, want to request a feature, or have a question or feedback, please send an email to [yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu). We love to hear from people using our code.

The code is published under the MIT License.