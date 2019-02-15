# variational Latent Gaussian Process

[![python 3.5](https://img.shields.io/badge/python-3.5-blue.svg?style=flat-square)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()

## Introduction

This repo contains the implementation of [variational Latent Gaussian Process (vLGP)](https://doi.org/10.1162/NECO_a_00953) 
([arXiv](https://arxiv.org/abs/1604.03053)) 
([video](https://youtu.be/CrY5AfNH1ik)) by 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)).
It has been developed with the goal of recovering dynamics from population spike trains. 

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

## Installation

#### pip
```bash
pip install vlgp
```

#### from source
```bash
git clone git@github.com:catniplab/vlgp.git
cd vlgp
pip install -e .
```

## Usage
To get started, please see the [tutorial](notebook/tutorial.ipynb).

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
