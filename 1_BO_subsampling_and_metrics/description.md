# Objective of this experiments

1. Warming up sensitivity for MV- LSSVM, this parameter if int then the number of elements to remove if p \in (0,1) the fraction to retain [p n].
2. Get subsampled distribution MMD, between themselves, full and the other methods. 
The subsampling methods are: 
- Full dataset 
- Nyström (related which is indeep random). 
- Kernel herding.
- Kernel clustering with (RFF)
- Indepedent Kernel Herding (our methods) 

Compare in the feature space.

3. Compare performance for LS-SVM. 

 
## To determine the hyperparameters 
Iterate over different sized and porcentajes.


1. BO, metric bal accuracy. 
Number of tries: 5 * dimension (heuristic )
Arguments:
    Subsampling methods `hyperparemeter
    - Number or percent of elements to subsample


space of search: 
    - gammas \in log([10^{5}, 1])* views (rbf hyperparameter)
    - C \in log([10^{5}, 10^5]) * views (this is the gamma in each LS-SVM regularization parameter)
    - rho  (couping parameter)
    - (only for Indepedent kernel herding) warming ratio \in [0,1] * views

