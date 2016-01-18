from injecting_bias import *
from scipy.stats import pearsonr, sem
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    
    # The percent symbol needs escaping in latex
    if plt.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'

def search_for_simpson(data, method, rand=None, bias=None, size=None):
    """ Search for example of simpsons paradox: a coefficient that has one sign when a 
    classifier is fit to all data, but a separate sign when fit to each genre separately."""
    if bias is None:
        train_x = data.train_x
        train_y = data.train_y
        train_c = data.train_c
    else:
        train_idx = make_confounding_data(data.train_x, data.train_y, data.train_c, .5, bias, size)
        train_x = data.train_x[train_idx]
        train_y = data.train_y[train_idx]
        train_c = data.train_c[train_idx]
        
    clf = method(train_x, train_y, train_c, rand, None)
    
    c1 = np.where(train_c == 1)
    c0 = np.where(train_c == 0)

    clf_c1 = method(train_x[c1], train_y[c1], train_c[c1], rand, None)
    clf_c0 = method(train_x[c0], train_y[c0], train_c[c0], rand, None)

    count = 0
    l = len(data.feature_names)
#     sum_coef = np.abs(clf.coef_[0][:l]) + np.abs(clf_c1.coef_[0][:l]) + np.abs(clf_c0.coef_[0][:l])
    paradoxes = []
    
    x2, pval = chi2(train_x, train_y)
    x2_sorted_idx = np.argsort(x2)[::-1]
    
    for i in x2_sorted_idx:
        # clf_c1 and clf_c0 have same sign and its opposite of clf's sign
        if clf_c1.coef_[0][i] * clf_c0.coef_[0][i] > 0 and clf_c1.coef_[0][i] * clf.coef_[0][i] < 0:
            count += 1
            paradoxes.append((data.feature_names[i],
                              clf.coef_[0][i],
                              clf_c1.coef_[0][i],
                              clf_c0.coef_[0][i],
                              train_x[c1, i].sum(),
                              train_x[c0, i].sum()))
    return count, paradoxes
#     print('Found %d paradoxes of %d features' % (count, len(data.feature_names)))
    
#     for p in paradoxes:
#         print('%20s\tall=%7.4f\thorror=%7.4f\tnon_horror=%7.4f\t#c1 =%4d\t#c0 =%4d' % p)

def simpson_paradox_count_bias(data, methods, size, rand=None, ntrials=10):
    results = {}
    for name, f in methods:
        biases = []
        spa_avg = []
        spa_sem = []
        for bias in np.arange(0.1, 1., 0.1):
            biases.append(bias)
            bias_spa_count = []
            for tr in range(ntrials):
                if tr == 0:
                    print(name, '- Bias =', bias)
                bias_spa_count.append(search_for_simpson(data, f, bias=bias, size=size, rand=rand)[0])
            spa_avg.append(np.mean(bias_spa_count))
            spa_sem.append(sem(bias_spa_count))
        results[name] = (spa_avg, spa_sem)
    return biases, results

def plot_spa_results(biases, results, markers, tofile=None, fmt='pdf', n_fts=None):
    fig, ax = plt.subplots()
    formatter = FuncFormatter(to_percent)
    plt.grid(True)
    ax.set_xlim([0., 1.])
    plt.xlabel('Bias')
    plt.ylabel('Simpson\'s paradox count')
    if n_fts is not None:
        plt.gca().yaxis.set_major_formatter(formatter)
        new_results = {}
        for name in list(results.keys()):
            spa_avg, spa_sem = [np.array(x) for x in results[name]]
            spa_avg /= n_fts
            spa_sem /= n_fts
            new_results[name] = (spa_avg, spa_sem)
            plt.ylabel('Percentage of features displaying Simpson\'s paradox')
        results = new_results
    for name in list(results.keys()):
        spa_avg, spa_sem = results[name]
        plt.errorbar(biases, spa_avg, yerr=spa_sem, label=name, fmt=markers[name])
    lgd = plt.legend(ncol=len(results), loc=3, mode='expand', bbox_to_anchor=(0., 1.02, 1., 0.),
                     borderaxespad=0., numpoints=2)

    if tofile is None:
        plt.show()
    else:
        plt.savefig(tofile, format=fmt, bbox_extra_artists=(lgd,), bbox_inches='tight')
