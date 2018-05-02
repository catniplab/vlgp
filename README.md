# variational Latent Gaussian Process

[![python 3.5](https://img.shields.io/badge/python-3.5-blue.svg?style=flat-square)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()

## Introduction

It contains the method of variational Latent Gaussian Process (vLGP) based on 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
It has been developed with the goal of recovering dynamics from population spike trains. 

## Changes
May 2, 2018
- Redesign data structure in an incremental way that keeps all trial-wise information (e.g. trial ID, stimuli and etc.)
- Allow trials to have different lengths
- Deprecate HDF5 format
- Optimize running time

2017
- New ```fit``` function now only requires observation and the number of latent.
- Save snapshots if *path* is passed to ```fit```.
- You can access the iterations via *callback*.

## Installation

```bash
pip install -e .
```

## Usage
To get started, please see the [tutorial](notebook/tutorial.ipynb).

The data are expected to be binned spike counts (Poisson) or/and LFP channels (Gaussian).
The required data is a list of trials. 
Each trial should be a dict that contains at least spike trains/LFP (key: ```"y"```) of shape (bin, neuron).

```python
import vlgp
trials = [{'y': y}]
result = vlgp.fit(trials, n_factors=2)  # 2D latent dyanmics
```

The function ```fit``` returns a dict that contains the latent dynamics, parameters and configuration. 

The default options are recommended for the purpose of stability but not necessarily optimal.
If any numerical error is raised, e.g. singular matrices, retry by changing the initial prior or other options.
 
## Modules

| module     | function                                                                                      |
|:-----------|-----------------------------------------------------------------------------------------------|
| api        | ```fit``` function                                                                            |
| core       | algorithm                                                                                     |
| math       | link functions, incomplelte Cholesky decompostion, angle between subspaces, orthogonalization |
| simulation | simulation of Gaussian process, Lorenz dynamics and spike trains                              |
| util       | lag matrix construction, rotation, load and save                                              |
