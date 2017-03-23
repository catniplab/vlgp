# README #

## Welcome

This package is written by Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])). 
It contains methods of variational Latent Gaussian Process (vLGP) model based on Yuan Zhao and Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
It has been developed and implemented with the goal of recovering dynamics from population spike trains. 

This package requires Python >= 3.5.

## Changes

- New *fit* function now only requires observation and the number of latent as argument, 
and returns a single dict containing all results. Usage: 
result = fit(y=y, lat_dim=2).
- It saves intermediate results if *path* is passed to *fit*.
- One can now add any function to the list argument *callbacks* to measure the iterations.

## Usage

To get started, see the examples in notebook: tutorial.ipynb.

The default options controlling algorithm are recommended for the purpose of stability but not necessarily the optimal.
If you encounter any numerical error (most likely singular matrix errors), try to change the prior and other options.

The data are expected to be spike counts (Poisson) or/and LFP channels (Gaussian) in time bins of shape (trial, bin, neuron).
 
## Modules

| module     | function                                                                                      |
|:-----------|-----------------------------------------------------------------------------------------------|
| api        | user interface                                                                                |
| core       | algorithm                                                                                     |
| ~~selection~~ | ~~model selection, CV, leave-one-out~~                                                     |
| gp         | hyperparameter optimization, kernel                                                           |
| math       | link functions, incomplelte Cholesky decompostion, angle between subspaces, orthogonalization |
| plot       | raster and dynamics                                                                           |
| simulation | simulation of Gaussian process, Lorenz dynamics and spike trains                              |
| util       | lag matrix construction, rotation, load and save                                              |

## Contact

If you notice a bug, want to request a feature, or have a question or feedback, please send an email to [yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu). We love to hear from people using our code.

The code is published under the MIT License.