# variational Latent Gaussian Process

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()
[![python 3.5](https://img.shields.io/badge/python-3.5-blue.svg?style=flat-square)]()
[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)]()
[![pip](https://img.shields.io/badge/install-pip-blue.svg?style=flat-square)]()

## Introduction

This repo contains the implementation of [variational Latent Gaussian Process (vLGP)](https://doi.org/10.1162/NECO_a_00953) 
([arXiv](https://arxiv.org/abs/1604.03053)) 
([video](https://youtu.be/CrY5AfNH1ik)) by 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)).
It has been developed with the goal of recovering low-dimensional dynamics from neural population recordings. 

## Installation

#### pip
```bash
pip install git+https://github.com/catniplab/vlgp.git
```

## Usage
The main entry is `vlgp.fit`. The `fit` function requires two arguments `trials` and `n_factors`. 
The former is expected as a *list* of *dictionaries*, each of which stores on trial and 
at least contains a identifier `ID` and the observation `y` in the shape of (bin, channel).
The later specifies the number of factors (latent processes).

```python
result = vlgp.fit(
    trials,       # list of dictionaries
    n_factors=3,  # dimensionality
)
```

The `fit` function returns a dictionary of `trials`, `params` and `config` as the fitted model.


Please see the [tutorial](notebook/tutorial.ipynb) for details.



## Citation
```
@Article{Zhao2017,
  author    = {Yuan Zhao and Il Memming Park},
  title     = {Variational Latent Gaussian Process for Recovering Single-Trial Dynamics from Population Spike Trains},
  journal   = {Neural Computation},
  year      = {2017},
  volume    = {29},
  number    = {5},
  pages     = {1293--1316},
  month     = {may},
  doi       = {10.1162/neco_a_00953},
  publisher = {{MIT} Press - Journals},
}
```

## Changes

2018

- New uniform data structure
- Support trials of unequal duration
- Faster
- Use NumPy data format

2017

- New ```fit``` function now only requires observation and the number of latent.
- Save snapshots if *path* is passed to ```fit```.
- You can access the iterations via *callback*.
