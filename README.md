# Learning-the-Optimal-Solution-Path (LSP)

Code associated with the paper "Beyond DescretizationL Learning the Optimal Solution Path".

LSP is an overall framework for learning the whole solution path for a family of hyperparametrized optimization problems. 

This implementation of LSP is based on [Pytorch Neural Network Modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) API.

## Overview

The code is divided into two folders:

- `lib` contains all of the files needed to run the method.
- `experiments` contains files concerning the experiments in the paper.

## Experiments

* `fair-regression` contains our main experiments associated with the paper. It runs a reweighted logistic regression on the highly imbalanced [law school admission Bar passage dataset](https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage?resource=download)
  
  - The `fair-regression/data` folder contains the dataset `bar_pass_prediction.csv` used for this experiment.
    
  - The `fair-regression/notebooks` folder contains the scripts written with Colab.
    
* `resularized-logit` contains some exploratory experiments. It runs a regularized logistic regression on the [Wisconsin breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer).
  
* `compare-basis` contains some exploratory experiments on the comparison between different basis functions.

