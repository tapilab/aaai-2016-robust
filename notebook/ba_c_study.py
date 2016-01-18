import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import chi2
from scipy import sparse
from scipy.stats import sem, pearsonr
from collections import defaultdict
from injecting_bias import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import scale


def do_c_study(c_range, filter_corr_diff, data, ntrials, rand, size, n):
    test_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    train_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    x2, pval = chi2(data.train_x, data.train_c)
    top_ft_idx = np.argsort(x2)[::-1][:n]
    print('%d top correlated features: %s' % (n, data.feature_names[top_ft_idx]))
    top_fts = list(zip(np.hstack([data.feature_names[top_ft_idx],
                                  ['c=0', 'c=1']]),
                       np.hstack([top_ft_idx,
                                  [-2, -1]])))
    c_values = []
    accuracies = defaultdict(list)
    coefs = defaultdict(lambda:defaultdict(list))
    
    for train_bias in train_biases:
        for test_bias in test_biases:
            for ti in range(ntrials):
                # Sample training and testing indices.
                test_idx = make_confounding_data(X=data.test_x, y=data.test_y, c=data.test_c,
                                                pos_prob=.5, bias=test_bias, size=size, rand=rand)  
                test_corr = pearsonr(data.test_y[test_idx], data.test_c[test_idx])[0]
                train_idx = make_confounding_data(X=data.train_x, y=data.train_y, c=data.train_c,
                                                  pos_prob=.5, bias=train_bias, size=size, rand=rand)   
                train_corr = pearsonr(data.train_y[train_idx], data.train_c[train_idx])[0]
                corr_diff = round(train_corr - test_corr, 1)
                if not filter_corr_diff(corr_diff):
                    continue
                if ti == 0:
                    #corr_diffs.append(corr_diff)
                    print('train_bias=', train_bias, 'train_corr=', train_corr,
                          'test_bias=', test_bias, 'test_corr=', test_corr,
                          'corr_diff=', corr_diff)
                    
                # Train and test each model.
                for c_val in c_range:
                    name = 'BA C=%f' % c_val
                    ba = backdoor_adjustment_var_C(data.train_x[train_idx], data.train_y[train_idx],
                                                   data.train_c[train_idx], rand, data.feature_names, c_val)
                    y_pred = ba.predict(data.test_x[test_idx])
                    y_true = data.test_y[test_idx]
                    accuracies[c_val].append(accuracy_score(y_true, y_pred))
                    ba_coefs = scale(ba.coef_[0])
                    for ft_name, ft_idx in top_fts:
                        coefs[ft_name][c_val].append(ba_coefs[ft_idx])
    return accuracies, coefs

def plot_c_study(c_range, accuracies_c, coefs, tofile=None, fmt='pdf'):
    fig, ax = plt.subplots()
    plot_handle_label = []
    # Compute values to plot
    acc_toplot = ([np.mean(np.abs(accuracies_c[c_val])) for c_val in c_range],
                  [sem(np.abs(accuracies_c[c_val])) for c_val in c_range])
    coefs_toplot = [(ft_name,
                     [np.mean(np.abs(coefs[ft_name][c_val])) for c_val in c_range],
                     [sem(np.abs(coefs[ft_name][c_val])) for c_val in c_range]) for ft_name in list(coefs.keys())]
    
    # Plot feature coefficient values
    ax2 = ax.twinx()
    grayscale = np.linspace(0, 1, len(coefs_toplot), endpoint=False)

    for gray, (ft_name, avg_coef, coef_err) in zip(grayscale, coefs_toplot):
        if ft_name == 'c=0':
            avg_c0, err_c0 = avg_coef, coef_err
            continue
        elif ft_name == 'c=1':
            avg_c1, err_c1 = avg_coef, coef_err
            continue
        else:
            continue
            ft_plt = ax2.errorbar(c_range, avg_coef, yerr=coef_err, c=repr(gray))
        plot_handle_label.append((ft_plt, ft_name))

    ft_plt_c0 = ax2.errorbar(c_range, avg_c0, yerr=err_c0, fmt='gx--')
    ft_plt_c1 = ax2.errorbar(c_range, avg_c1, yerr=err_c1, fmt='rx--')
    plot_handle_label.extend([(ft_plt_c0, 'coef $c_0$'), (ft_plt_c1, 'coef $c_1$')])

    # Plot accuracy line
    acc_plt = ax.errorbar(c_range, *acc_toplot, fmt='ks-')
    plot_handle_label.append((acc_plt, 'Accuracy'))
    ax.set_xscale('log')
    ax.grid(True)
    ax.set_xlabel('Value of $v_1$')
    ax.set_ylabel('Accuracy')

    # Plot c=0 and c=1 features coeffs.
    
    ax2.set_xscale('log')
    ax2.set_ylabel('Absolute coefficient value')

    lgd = ax.legend(*zip(*plot_handle_label), ncol=len(plot_handle_label), loc=3, mode='expand',
                     bbox_to_anchor=(0., 1.02, 1., 0.), borderaxespad=0., numpoints=2)
    if tofile is None:
        plt.show()
    else:
        plt.savefig(tofile, format=fmt)
