import numpy as np
import matplotlib.pyplot as plt
from injecting_bias import *
from sklearn.preprocessing import scale
from collections import defaultdict

def changing_coef_plot_given_idx(data, models, biases, size, indices, trials=10, title=None, class_labels=[0,1], feature_names=None, tofile=None, fmt='pdf'):
    if len(models) < 2:
        raise Exception('Models vector should be of length at least 2.')
    rand = np.random.RandomState(11111991)
    for t in models:
        if len(t) != 3:
            raise Exception('Each model should be a tuple containing (model function, model name, marker)')

    if feature_names is None:
        feature_names = data.feature_names

    for tr_bias in biases:
        coeff_diffs = []
        cums = defaultdict(list)
        print("bias = %.1f" % tr_bias)
        for ti in range(trials):
            tr_idx = make_confounding_data(data.train_x, data.train_y, data.train_c, .5, tr_bias, size,
                                           rand=rand)
            for model, name, marker in models:
                clf = model(data.train_x[tr_idx], data.train_y[tr_idx], data.train_c[tr_idx], None, None)
                cums[name].append(clf.coef_[0])
        avg_cums = {}
        for name in list(cums.keys()):
            avg_cums[name] = np.mean(cums[name], axis=0)
        
        n = len(indices)
        Y = np.arange(n)

        fig, ax = plt.subplots(figsize=(6,int(.4*n)))
        
        # vertical line
        line_Y = np.arange(-1,n+2)
        plt.plot([0]*(line_Y.shape[0]), line_Y, '--', color='k')

        # # data points
        Xs = []
        for model, name, marker in models:
            X = avg_cums[name][indices]
            Xs.append(X)
            plt.plot(X, Y, marker, label=name)

        # # arrows
        for i in range(len(Xs)-1):
            X1 = Xs[i]
            X2 = Xs[i+1]
            diffs = X1-X2
            for i, d in enumerate(diffs):
                sign_arrow = np.sign(d)
                ax.annotate("", (X2[i], i), (X1[i], i),
                            arrowprops=dict(arrowstyle="->", fc='w', ec='k'))

        # legend arrows
        for c_label, xlim in zip(class_labels, ax.get_xlim()):
            ax.annotate("", (xlim, n), (0, n),
                        arrowprops=dict(arrowstyle='-|>', linestyle='dashed', fc='k', ec='k'))
            s = str(c_label)
            ax.text(xlim/2., n+0.5, s, horizontalalignment='center', verticalalignment='center', fontsize=10)

        # legend and setting
        #lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #                 mode="expand", ncol=len(models), numpoints=1, borderaxespad=0.)
        ax.set_ylim([-1, n+1])
        ax.set_yticks(np.arange(0, n))
        ax.set_yticklabels(feature_names[indices])
        ax.set_xlabel('Coefficient value')
        if title:
            ax.set_title(title)
        #plt.grid(True)
        ax.xaxis.grid(True)
        [line.set_zorder(3) for line in ax.lines]
        if tofile is None:
            plt.show()
        else:
            fig.tight_layout()
            plt.savefig(tofile, format=fmt, bbox_inches='tight', bbox_extra_artists=(lgd,))

def changing_coef_plot(data, models, biases, size, transformation, trials=10, n=10, title=None, class_labels=[0,1], feature_names=None):
    if len(models) != 2:
        raise Exception('Models vector should be of length 2.')
    rand = np.random.RandomState(11111991)
    for t in models:
        if len(t) != 3:
            raise Exception('Each model should be a tuple containing (model function, model name, marker)')

    if feature_names is None:
        feature_names = data.feature_names
            
    model1, label1, shape1 = models[0]
    model2, label2, shape2 = models[1]

    for tr_bias in biases:
        coeff_diffs = []
        cum1 = []
        cum2 = []
        print("bias = %.1f" % tr_bias)
        for ti in range(trials):
            tr_idx = make_confounding_data(data.train_x, data.train_y, data.train_c, .5, tr_bias, size,
                                           rand=rand)
            clf1 = model1(data.train_x[tr_idx], data.train_y[tr_idx], data.train_c[tr_idx], None, None)
            clf2 = model2(data.train_x[tr_idx], data.train_y[tr_idx], data.train_c[tr_idx], None, None)
            if clf1.coef_.shape == clf2.coef_.shape:
                cum1.append(clf1.coef_[0])
                cum2.append(clf2.coef_[0])
            elif clf1.coef_.shape[1] < clf2.coef_.shape[1]:
                diff_len = clf1.coef_.shape[1] - clf2.coef_.shape[1]
                cum1.append(clf1.coef_[0])
                cum2.append(clf2.coef_[0][:diff_len])
            else:
                diff_len = clf2.coef_.shape[1] - clf1.coef_.shape[1]
                cum2.append(clf2.coef_[0])
                cum1.append(clf1.coef_[0][:diff_len])

        X1, X2, coeff_idx = transformation(cum1, cum2, n)
        
        Y = np.arange(n)

        fig, ax = plt.subplots(figsize=(6,int(.4*n)))
        
        # vertical line
        line_Y = np.arange(-1,n+2)
        plt.plot([0]*(line_Y.shape[0]), line_Y, '--', color='k')
        
        # data points
        h1, = plt.plot(X1, Y, shape1, label=label1)
        h2, = plt.plot(X2, Y, shape2, label=label2)

        # arrows
        diffs = X1-X2
        for i, d in enumerate(diffs):
            sign_arrow = np.sign(d)
            ax.annotate("", (X2[i], i), (X1[i], i),
                        arrowprops=dict(arrowstyle="->", fc='w', ec='k'))

        for c_label, xlim in zip(class_labels, ax.get_xlim()):
            ax.annotate("", (xlim, n), (0, n),
                        arrowprops=dict(arrowstyle='-|>', linestyle='dashed', fc='k', ec='k'))
            s = str(c_label)
            ax.text(xlim/2., n+0.5, s, horizontalalignment='center', verticalalignment='center', fontsize=10)

        # legend and setting
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   mode="expand", ncol=i+1, numpoints=1, borderaxespad=0.)
        ax.set_ylim([-1, n+1])
        ax.set_yticks(np.arange(0, n))
        ax.set_yticklabels(feature_names[coeff_idx])
        ax.set_xlabel('Coefficient value')
        ax.set_ylabel('Feature names')
        if title:
            ax.set_title(title)
        #plt.grid(True)
        ax.xaxis.grid(True)
        [line.set_zorder(3) for line in ax.lines]
        plt.show()

def most_changing_coef(cum1, cum2, n):
    mean_coef1 = np.array(cum1).mean(axis=0)
    mean_coef2 = np.array(cum2).mean(axis=0)
    mean_diffs = np.abs(mean_coef1 - mean_coef2)

    coeff_idx = np.argsort(mean_diffs)[::-1][:n][::-1]
    X1 = mean_coef1[coeff_idx]
    X2 = mean_coef2[coeff_idx]
    return X1, X2, coeff_idx

def most_changing_sign_coef(cum1, cum2, n):
    mean_coef1 = np.array(cum1).mean(axis=0)
    mean_coef2 = np.array(cum2).mean(axis=0)
    changing_signs = np.sign(mean_coef1 * mean_coef2)
    coeff_diffs = mean_coef1 - mean_coef2
    coeff_diffs[np.where(changing_signs == 1)] = 0
    coeff_idx = np.argsort(np.abs(coeff_diffs))[::-1][:n][::-1]
    X1 = mean_coef1[coeff_idx]
    X2 = mean_coef2[coeff_idx]
    
    return X1, X2, coeff_idx
