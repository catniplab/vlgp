# variational Latent Gaussian Process

[![python 3.5](https://img.shields.io/badge/python-3.5-blue.svg?style=flat-square)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()

## Introduction

It contains the method of variational Latent Gaussian Process (vLGP) based on 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
It has been developed with the goal of recovering dynamics from population spike trains. 

## Changes

- New *fit* function now only requires observation and the number of latent as argument, 
and returns a single dict containing all results. Usage: 
result = fit(y=y, lat_dim=2).
- It saves intermediate results if *path* is passed to *fit*.
- One can now add any function to the list argument *callbacks* to measure the iterations.

## Usage

To get started, see the examples in the [tutorial](notebook/tutorial.ipynb).

The default options controlling algorithm are recommended for the purpose of stability but not necessarily the optimal.
If you encounter any numerical error (most likely singular matrix errors), try to change the prior and other options.

The data are expected to be spike counts (Poisson) or/and LFP channels (Gaussian) in time bins of shape (trial, bin, neuron).
 
## Modules

| module     | function                                                                                      |
|:-----------|-----------------------------------------------------------------------------------------------|
| api        | main entry                                                                                    |
| core       | algorithm                                                                                     |
| math       | link functions, incomplelte Cholesky decompostion, angle between subspaces, orthogonalization |
| simulation | simulation of Gaussian process, Lorenz dynamics and spike trains                              |
| util       | lag matrix construction, rotation, load and save                                              |
