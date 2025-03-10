# Learning-the-Optimal-Solution-Path (LSP)

Code associated with the paper "Beyond Discretization: Learning the Optimal Solution Path".

LSP is an overall framework for learning the whole solution path for a family of hyperparametrized optimization problems. 

This implementation of LSP is based on [Pytorch Neural Network Modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) API.

We compare LSP with a naive grid search method.

## Overview

- `lib` contains all of the files needed to run LSP and naive grid search.
- `data` contains data files concerning the experiments in the paper, including both input datasets and results.
- `notebooks` contain all scripts and outputs for the experiments.

## Experiments

We carry out two sets of experiments: `reweighted-logistic-regression` and `portfolio-allocation`.

* `reweighted-logistic-regression` runs a reweighted logistic regression on the highly imbalanced [law school admission Bar passage dataset](https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage?resource=download)
   
  - We use two kinds of polynomial bases for LSP: Legendre polynomials and Jacobi polynomials. The results are contained in `reweighted-logistic-regression/laguerre` and `reweighted-logistic-regression/legendre` folders respectively.

  - The loss function runs on a single hyperparameter (1-d).
      
  - The `data/eweighted-logistic-regression` folder contains the dataset `bar_pass_prediction.csv` used for this experiment.
    

 
    
  
* `portfolio-allocation` runs a portfolio allocation problem: $\min_\theta \lambda_1\theta^\top \Sigma \theta - \lambda_2 \mu^\top \theta + \sum_i (\theta_i^2 + .01^2)^{1/2} - .01$, where $\Sigma$ is the covariance matrix of the `10_Industry_Portfolios_10_Year_Monthly.csv` dataset, and $\mu$ is the expected return of the same dataset. 

  - We use a bivariate Legendre polynomial basis for LSP.
 
  - The loss function runs on 2 hyperparameters (2-d).
 
  - The `high-dimension` folder contains a variation of the portfolio allocation problem that considers the objective $h(\theta, \lambda) =  -\lambda_1 \cdot \mu^\top \theta + \lambda_2 \cdot \theta^\top \Sigma \theta + \|\theta -\lambda_{3:12}\|_2^2$, so that the hyperparameter $\lambda$ is 12-dimensional.
    
  - The `data/portfolio-allocation` folder contains the dataset `10_Industry_Portfolios_10_Year_Monthly.csv`, `decomp_cov.csv` and `mean.csv` used for this experiment. `decomp_cov.csv` and `mean.csv` are the covariance matrix and expected return computed beforehand.

