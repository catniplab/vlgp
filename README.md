# README #

## Welcome

This package is written by Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])). 
It contains methods of variational Latent Gaussian Process (vLGP) model based on Yuan Zhao and Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
It has been developed and implemented with the goal of recovering dynamics from population spike trains. 

The code is written in Python 3. It needs further testing and is subject to change.

## Usage

To get started, see the examples in notebook: ./tutorial.ipynb. 
There are 5 Lorenz simulations and 5 LDS simulations in './data' for the tutorial. The corresponding inference and cross-validation results are in './output'. 
You may skip the code in tutorial and load them directly to save time.

The default options controlling algorithm are recommended for the purpose of stability but not necessarily the optimal.
If you encounter any numerical error (most likely singular matrix errors), try to change the prior and enable the Hessian adjustment.

This package heavily depends on NumPy. All nonscalar data are expected to be in ndarray-compatible type. 

The data for training models are expected to be spike trains (LFP channels will be supported in future). 
The spike trains should be binned and shaped into array as (trial, bin, neuron) before passing to the functions.
Each element of array is the spike count in that time bin of certain neuron and trial.

## Modules

| module     | function                                                                                      |
|:-----------|-----------------------------------------------------------------------------------------------|
| core       | model fitting, prediction                                                                     |
| selection  | model selection, CV, leave-one-out                                                            |
| hyper      | hyperparameter optimization, kernel                                                           |
| math       | link functions, incomplelte Cholesky decompostion, angle between subspaces, orthogonalization |
| plot       | raster and dynamics                                                                           |
| simulation | simulation of Gaussian process, Lorenz dynamics and spike trains                              |
| util       | lag matrix construction, rotation, load and save                                              |

## Contact

If you notice a bug, want to request a feature, or have a question or feedback, please send an email to [yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu). We love to hear from people using our code.

The code is published under the MIT License.