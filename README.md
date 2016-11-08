# Inductive Bias of Deep Convolutional Networks through Pooling Geometry
This repository contains scripts for reproducing the experiments in the paper [Inductive Bias of Deep Convolutional Networks through Pooling Geometry](https://arxiv.org/abs/1605.06743). 

## Disclaimer
Due to inherent randomness in the experiments, obtained results might slightly deviate from those reported in the paper.
Qualitatively however, full compliance with the paper is expected.

## Installation
The repository includes as a submodule the [Caffe fork of the SimNets Architecture](https://github.com/HUJI-Deep/caffe-simnets).
To clone the repository along with the submodule, call `git clone` with its `--recursive` flag:
```
git clone --recursive https://github.com/HUJI-Deep/inductive-pooling.git
```
After cloning, make sure to install the Python package versions listed in requirements.txt:
```
pip install -r requirements.txt
```
Finally, go to the SimNets submodule in `deps/simnets` and follow its [installation instructions](https://github.com/HUJI-Deep/caffe-simnets).

## Running the Experiments
The experiments in the paper are based on a synthetic classification benchmark consisting of images displaying random shapes ('blobs').
Two labels are assigned to images, ranking properties of the displayed blob: 
* **closure**: how morphologically closed the blob is
* **symmetry**: how left-right symmetric the blob is about its center
Convolutional networks with two pooling geometries are evaluated: 
* **square**: standard contiguous square windows
* **mirror**: pooling windows join together nodes with their spatial reflections
It is shown that different pooling geometries lead to superior performance in different tasks.
Specifically, square pooling outperforms mirror on the task of closure ranking, whereas the opposite occurs when ranking symmetry.

Scripts for running the experiments are under `exp/blob`.
`generate_blob.py` can be called directly to generate the dataset, though this is not necessary (running a convolutional network will generate the dataset automatically in case it is missing).

The folder `exp/blob/cac` contains scripts for evaluating *convolutional arithmetic circuits* (convolutional networks with linear activation and product pooling):
* **net.prototmp**: template for prototxt specifying network architecture and training scheme
* **train_plan.json**: list of values for the unspecified parameters in `net.prototmp`
* **train.py**: substitutes the values in `train_plan.json` into `net.prototmp` and trains the network, measuring its train and test accuracies along the way
By default, calling `train.py` initiates 4 runs, corresponding to the different combinations of closure/symmetry task and square/mirror pooling.
The gaps in performance (square pooling better than mirror for closure, opposite for symmetry) should be evident.

To evaluate *convolutional rectifier networks* (convolutional networks with ReLU activation and max or average pooling), go to `exp/blob/crn`.
This folder contains the exact same files as `exp/blob/cac`, implementing the exact same functionalities.
The only difference (from the user's perspective) is that calling `train.py` initiates 8 runs instead of 4, corresponding to the fact that every setting is run twice -- once with average pooling, and once with max pooling.

You are welcome to alter the architectural and optimization parameters of the networks.
In particular, network breadth (number of channels in hidden layers) can be configured in `train_plan.json`, and network depth can be set by modifying `net.prototmp`.

###### Important Note
Please do not move any of the scripts mentioned above, as they all rely on their original path relative to the root of the repository.