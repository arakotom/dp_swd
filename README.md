# Differentially Private Sliced Wasserstein Distance

This repository contains the code of our paper  "Differentially Private Sliced Wasserstein Distance" to appear at ICML 2021.
We will make it available asap and before the conference starts. 

## Installation

* Pytorch 1.8

## Short Summary

DP-SWD computation is in file distrib_distance.py
ClassDANN and ClassSWD contains the DP-DANN and the DP-SWD train/inference algorithm
da_settings.py and da_models are utility files that contain model and learning parameters.

# Results reproduction

for reproducing the Domain adaptation results, one has to run the da_digits.py file. Selecting the settings allows to train either MNIST-USPS or USPS-MNIST.

# Citation 



