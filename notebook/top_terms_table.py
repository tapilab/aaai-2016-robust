import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from scipy import sparse
from scipy.stats import sem, pearsonr
from collections import defaultdict
from injecting_bias import *


def do_top_coef_table(data, model, ntrials, size, n, rand, perc=0.1, feature_names=None):
    if feature_names is None:
        feature_names = data.feature_names
    train_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

    # add a star when printing one of the top correlated feature in the feature with the highest coeffs
    x2, pval = chi2(data.train_x, data.train_c)
    n_top_corr_fts = round(data.train_x.shape[1]*perc/100)
    top_ft_idx = np.argsort(x2)[::-1][:n_top_corr_fts]
    top_ft_names = data.feature_names[top_ft_idx]
    print("Top %.2f %% of features that are the most correlated with the confounder:\n%s" % (perc, top_ft_names))
    train_corrs = set()
    coefs = defaultdict(list)
    
    for train_bias in train_biases:
        print(train_bias)
        for ti in range(ntrials):
            # Sample training and testing indices.
            train_idx = make_confounding_data(X=data.train_x, y=data.train_y, c=data.train_c,
                                              pos_prob=.5, bias=train_bias, size=size, rand=rand)  
            train_corr = round(pearsonr(data.train_y[train_idx], data.train_c[train_idx])[0], 1)
            train_corrs.add(train_corr)
            # Train and test each model.
            clf = model(data.train_x[train_idx], data.train_y[train_idx],
                        data.train_c[train_idx], rand, data.feature_names)

            coefs[train_corr].append(clf.coef_[0])
            
    for train_corr in sorted(list(train_corrs)):
        print("Train correlation:", train_corr)
        avg_coefs = np.mean(coefs[train_corr], axis=0)
        sorted_idx = np.argsort(np.abs(avg_coefs))[::-1][:n]
        top_feature_names = feature_names[sorted_idx]
        top_feature_coefs = avg_coefs[sorted_idx]
        for ft_name, ft_coef in zip(top_feature_names, top_feature_coefs):
            if ft_name in top_ft_names:
                ft_name += '*'
            print('\t%20s\t%.3f' % (ft_name, ft_coef))
