import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, sem
from collections import defaultdict

def plot_coef(all_coef, all_corrs, confounding_feature, feature_names):
    # Plot learned coefficients as correlation varies.
    plt.figure()
    colors = ['k', 'r', 'b', 'g', 'c']
    for ci, name in enumerate(all_coef):
        coef = all_coef[name]
        corrs = all_corrs[ci]
        coef_means = [np.mean(x) for x in coef]
        plt.errorbar(corrs, coef_means, yerr=[sem(x) for x in coef],
                     fmt=colors[ci] + 'o', label=name + '_' + feature_names[confounding_feature])
    plt.ylabel('coefficient (standardized)')
    plt.xlabel('correlation(y, c)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=len(all_corrs))
    plt.show()
    
def plot_matrix(mat, title, labels_x=None, labels_y=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat)
    plt.title(title)
    fig.colorbar(cax)
    if labels_x:
        ax.set_xticks(np.arange(len(labels_x)))
        ax.set_xticklabels(labels_x)
    if labels_y:
        ax.set_yticks(np.arange(len(labels_y)))
        ax.set_yticklabels(labels_y)
    plt.xlabel('Train bias')
    plt.ylabel('Test bias')
    plt.show()

def plot_difficulties(difficulties, bins=10):
    # Data   
    plot_data = []
    names = []
    for y_true, c_val in [(0,0), (0,1), (1,0), (1,1)]:
        diff_yc = difficulties[2*y_true+c_val]
        plot_data.append(diff_yc)
        names.append('y=%d, c=%d' %(y_true, c_val))
        print("y=%d, c=%d, mean=%.5f, std=%.5f" % (y_true, c_val, np.mean(diff_yc), np.std(diff_yc)))

    # Boxplots
    fig, axes = plt.subplots()
    plt.boxplot(plot_data)
    xtickNames = plt.setp(axes, xticklabels=names)
    axes.set_ylim([-.01, 1.01])
    axes.set_ylabel('Difficulty')
    plt.show()

    # Histogram
    fig, axes = plt.subplots()
    plt.yscale('log', nonposy='clip')
    hist = plt.hist(plot_data, label=names, bins=bins)
    plt.legend()
    axes.set_xlabel('Difficulty')
    axes.set_ylabel('Count (log-scale)')
    plt.show()

def plot_all_accuracies(all_accuracies, x_axis_tuple, title='Accuracy', xlim=None, keys=None, train_bias=None):
    x_idx_sorted = np.argsort(x_axis_tuple[0])
    x_axis_values, x_axis_key_name, x_axis_title = x_axis_tuple
    x_axis_values = sorted(x_axis_values)

    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(14, 8))
    f.text(0.5,0.98,title,horizontalalignment='center',verticalalignment='top', fontsize=16)
    f.text(0.5,0.07, x_axis_title,horizontalalignment='center',verticalalignment='bottom', fontsize=14)
    markers = ['s', 'v', 'x', 'D', '^', '*']
    method_keys = list(all_accuracies[0].keys())
    if keys is not None:
        method_keys = [k for k in keys if k in method_keys]
    method_keys = sorted(method_keys)
    yc_accuracies_tuples = []
    # build all accuracy dictionaries
    for y in range(3):
        for c in range(3):
            k = 3*y+c
            accuracies_yc = all_accuracies[k]

            ls = []
            first = True
            for ci, name in enumerate(method_keys):
                ax_title = "%s %s" % ('y=%d' % (y) if y != 2 else '',
                                      'c=%d' % (c) if c != 2 else '')
                accuracy_dicts = accuracies_yc[name]
                #accuracies = [[item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb] for tb in x_axis_values]
                accuracies = np.array([[item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb and train_bias in [None, item['train_bias']]] for tb in x_axis_values])

                acc_means = [np.mean(x) for x in accuracies]
                axarr[y,c].set_title(ax_title)
                e = axarr[y,c].errorbar(x_axis_values, acc_means, yerr=[sem(x) for x in accuracies], label=str(name), marker=markers[ci])
                if first:
                    ls.append(e)
                if xlim:
                    axarr[y,c].set_xlim(xlim)
            first = False

    f.legend(ls, method_keys, ncol=len(method_keys), loc='lower center')
    plt.show()
    
def plot_accuracy(all_accuracies, x_axis_tuple, y, c, title=None, xlim=None, keys=None, train_bias=None):
    x_axis_values = np.array(x_axis_tuple[0])
    x_axis_key_name = x_axis_tuple[1]
    x_axis_title = x_axis_tuple[2]
    if title is None:
        title = 'Accuracy for y=%d, c=%d' % (y, c)
    markers = ['s', 'v', 'x', 'D', '^', '*']
    method_keys = list(all_accuracies[0].keys())
    if keys is not None:
        method_keys = [k for k in keys if k in method_keys]
    method_keys = sorted(method_keys)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.set_position([0.1,0.1,0.85,0.75])
    plt.title(title)
    k = 3*y+c
    accuracies_yc = all_accuracies[k]
    for ci, name in enumerate(method_keys):
        accuracy_dicts = accuracies_yc[name]     
        idx_sorted = np.argsort(x_axis_values)
        accuracies = np.array([[item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb and train_bias in [None, item['train_bias']]] for tb in x_axis_values])
        #accuracies = np.array([[item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb] for tb in x_axis_values])
        acc_means = np.array([np.mean(x) for x in accuracies])
        e = ax.errorbar(x_axis_values[idx_sorted],
                        acc_means[idx_sorted], yerr=[sem(x) for x in accuracies[idx_sorted]],
                        marker=markers[ci], ls='-', label=str(name))
        if xlim:
            ax.set_xlim(xlim)
    ax.legend(ncol=len(method_keys), loc='upper center', bbox_to_anchor=(0.5, 1.18),
              fancybox=True, borderpad=0.5)
    plt.show()

def plot_subsampling_expmt(X, Y, to_plot=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ['b', 'g', 'r', 'c', 'y', 'm']
    for i, (model_name, scores_list) in enumerate(Y.items()):
        if to_plot is None or model_name in to_plot:
            mean_scores = [np.mean(x) for x in scores_list]
            yerr = [sem(x) for x in scores_list]
            plt.errorbar(X, mean_scores, yerr=yerr, label=model_name, marker='o', color=colors[i])
    plt.legend(loc='lower right')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt.show()

def export_plot_accuracy(filename, all_accuracies, x_axis_tuple, y, c, fmt='pdf', title=None, xlim=None, keys=None, mask=None, train_bias=None,
                         legends=None, xlabel=None, ylabel=None, set_xticks=None, ncol=None):
    x_axis_values = np.array(x_axis_tuple[0])
    x_axis_key_name = x_axis_tuple[1]
    x_axis_title = x_axis_tuple[2]
    if title is None:
        title = 'Accuracy for y=%d, c=%d' % (y, c)
    markers = ['s', 'v', 'x', 'D', '^', '*']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    method_keys = np.array(sorted(list(all_accuracies[0].keys()))) if keys is None else np.array(keys)
    sorted_keys = np.argsort(method_keys)
    method_keys = method_keys[sorted_keys]
    method_mask = None if mask is None else np.array(mask)[sorted_keys]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(True)
    #ax.set_position([0.15,0.1,1.,0.75])
    plt.title(title)
    k = 3*y+c
    accuracies_yc = all_accuracies[k]
    for ci, name in enumerate(method_keys):
        if mask is not None and not method_mask[ci]:
            continue
        accuracy_dicts = accuracies_yc[name]        
        idx_sorted = np.argsort(x_axis_values)
        x_sorted = sorted(set(x_axis_values))
        accuracies = np.array([[item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb and train_bias in [None, item['train_bias']]] for tb in x_sorted])
        acc_means = np.array([np.mean(x) for x in accuracies])
        acc_yerr = np.array([sem(x) for x in accuracies])
        e = ax.errorbar(x_sorted, acc_means, yerr=acc_yerr,
                        marker=markers[ci], c=colors[ci], ls='-', label=str(name) if legends is None else legends[ci])
        if xlim:
            ax.set_xlim(xlim)
    if ncol is None:
        ncol = ci+1
    lgd = ax.legend(ncol=ncol, loc=3, mode='expand', bbox_to_anchor=(0., 1.02, 1., 0.),
                    borderaxespad=0., numpoints=2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if set_xticks is not None:
        ax.set_xticks(set_xticks)
    # put grid behind other elements
    [line.set_zorder(3) for line in ax.lines]
    fig.savefig(filename, format=fmt, bbox_extra_artists=(lgd,), bbox_inches='tight')


def get_table_accuracy(all_accuracies, x_axis_tuple, y, c, keys=None, mask=None):
    x_axis_values = np.array(x_axis_tuple[0])
    x_axis_key_name = x_axis_tuple[1]
    x_axis_title = x_axis_tuple[2]

    method_keys = np.array(sorted(list(all_accuracies[0].keys()))) if keys is None else np.array(keys)
    sorted_keys = np.argsort(method_keys)
    method_keys = method_keys[sorted_keys]
    method_mask = None if mask is None else np.array(mask)[sorted_keys]

    k = 3*y+c
    accuracies_yc = all_accuracies[k]

    accuracies_name = defaultdict(list)
    accuracies_tb = defaultdict(dict)
    avg_accuracies = {}
    sorted_x = sorted(set(x_axis_values))
    for ci, name in enumerate(method_keys):
        accuracy_dicts = accuracies_yc[name]        
        for tb in sorted_x:
            vals = [item['acc'] for item in accuracy_dicts if item[x_axis_key_name] == tb]
            mean_vals = np.mean(vals)
            accuracies_name[name].append(mean_vals)
            #print(tb, mean_vals)
            accuracies_tb[tb][name] = mean_vals
        avg_accuracies[name] = np.mean(accuracies_name[name])

    for tb in sorted_x:
        results = []
        for name in method_keys:
            val = np.round(accuracies_tb[tb][name], 4)
            results.append(val)
        m = max(results)
        new_results = ["%.4f" % r if r != m else "\\textbf{%.4f}" % r for r in results]
        print("\\textbf{%.1f} & " % tb, " & ".join(new_results), "\\\\")
    results = []
    for name in method_keys:
        val = np.round(avg_accuracies[name], 4)
        results.append(val)
    m = max(results)
    new_results = ["\\textit{%.4f}" % r if r != m else "\\textbf{\\textit{%.4f}}" % r for r in results]
    print("\\textbf{\\textit{mean}} & ", " & ".join(new_results), "\\\\")
    return avg_accuracies
    
