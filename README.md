# Differentially Private Sliced Wasserstein Distance

This repository contains the code of our paper  "Differentially Private Sliced Wasserstein Distance" to appear at ICML 2021.

## Installation

* Pytorch 1.8
* autodp

## Short Summary

DP-SWD computation is in file distrib_distance.py
ClassDANN and ClassSWD contains the DP-DANN and the DP-SWD train/inference algorithm
da_settings.py and da_models are utility files that contain model and learning parameters.

the file da_dp_analysis.py presents how we have computed the noise standard deviation given a desired accuracy. For this, we use the autodp package https://github.com/yuxiangw/autodp

# Results reproduction

for reproducing the Domain adaptation results, one has to run the da_digits.py file. Selecting the settings allows to train either MNIST-USPS or USPS-MNIST.

# Citation 

if you use this code for your research, please cite our work


@InProceedings{pmlr-v139-rakotomamonjy21a,
  title = 	 {Differentially Private Sliced Wasserstein Distance},
  author =       {Rakotomamonjy, Alain and Ralaivola, Liva},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8810--8820},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/rakotomamonjy21a/rakotomamonjy21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/rakotomamonjy21a.html}
}


