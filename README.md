# variational Latent Gaussian Process

[![python 3.5](https://img.shields.io/badge/python-3.5-blue.svg?style=flat-square)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)]()

## Introduction

It contains the method of variational Latent Gaussian Process (vLGP) based on 
Yuan Zhao ([yuan.zhao@stonybrook.edu](yuan.zhao@stonybrook.edu])) and 
Il Memming Park's ([memming.park@stonybrook.edu](memming.park@stonybrook.edu)) work.
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

```bash
git clone git@github.com:catniplab/vlgp.git
cd vlgp
pip install -e .
```

## Usage
To get started, please see the [tutorial](notebook/tutorial.ipynb).
 