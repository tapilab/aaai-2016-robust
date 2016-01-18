def sample_from_difficulties(d, size):
    sorted_d = np.sort(d)
    sorted_d_idx = np.argsort(d)
    sampling = []
    i = 0
    while i < size:
        s_difficulty = np.random.normal(data.d_mean, data.d_std)
        if s_difficulty < 0 or s_difficulty > 1:
            continue
        i += 1
        closest_instance = bisect.bisect_left(sorted_d, s_difficulty)
        closest_instance_idx = sorted_d_idx[closest_instance]
        sampling.append(closest_instance_idx)
        sorted_d = np.delete(sorted_d, closest_instance)
        sorted_d_idx = np.delete(sorted_d_idx, closest_instance)
    return sampling


def make_confounding_data_from_difficulties(X, y, c, d, pos_prob, bias, size, rand, plotting=False):
    """ Create Sample a dataset of given size where c is a confounder for y with strength=bias.
        We take care not to introduce selection bias (that is, p(c=1) is representative of training data).
        This assumes that #[c=1] < #[y=1].
        
        X: data matrix
        y: labels (0, 1)
        c: confounding labels (0,1)
        d: difficulty values (in [0,1])
        pos_prop: proportion of instances where y=1
        bias: amount of bias (0-1)
        size: number of samples
        rand: RandomState
    """
    both_pos = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 1])
    both_neg = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 0])
    ypos_cneg = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 0])
    yneg_cpos = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 1])


    # if bias=.9, then 90% of instances where c=1 will also have y=1
    # similarly, 10% of instances where c=1 will have y=0
    cprob = 1. * sum(c) / len(c)
    n_cpos = int(cprob * size)
    n_cneg = size - n_cpos
    n_ypos = int(pos_prob * size)
    n_yneg = size - n_ypos
#     print(n_cpos,n_cneg, n_ypos, n_yneg)
    n_11 = int(bias * n_cpos)
    n_01 = int((1 - bias) * n_cpos)
    n_10 = n_ypos - n_11
    n_00 = n_yneg - n_01  
#     print(n_11, n_01, n_10, n_00)  
    
    s_11 = sample_from_difficulties(d[both_pos], n_11)
    s_00 = sample_from_difficulties(d[both_neg], n_00)
    s_10 = sample_from_difficulties(d[ypos_cneg], n_10)
    s_01 = sample_from_difficulties(d[yneg_cpos], n_01)

    if plotting:
        diff_plot = {0: d[both_neg[s_00]], 1: d[yneg_cpos[s_01]], 2: d[ypos_cneg[s_10]], 3: d[both_pos[s_11]]}
        plot_difficulties(diff_plot)
    
    r = np.hstack([both_pos[s_11], both_neg[s_00], ypos_cneg[s_10], yneg_cpos[s_01]])
    return r


def do_confounding_trials_with_difficulty(models, data, ntrials, rand):  
    test_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    train_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    corr_diffs = []
    accuracies = defaultdict(lambda: defaultdict(lambda: []))
    idx_horror = np.where(data.feature_names == 'horror')
    for train_bias in train_biases:
        for test_bias in test_biases:
            for ti in range(ntrials):
                # Sample training and testing indices.
                test_idx = make_confounding_data_from_difficulties(X=data.test_x, y=data.test_y, c=data.test_c, d=data.test_d,
                                                                   pos_prob=.5, bias=test_bias, size=1500, rand=rand)  
                test_corr = pearsonr(data.test_y[test_idx], data.test_c[test_idx])[0]
                train_idx = make_confounding_data_from_difficulties(X=data.train_x, y=data.train_y, c=data.train_c,
                                                                    d=data.train_d, pos_prob=.5, bias=train_bias, size=1500,
                                                                    rand=rand)   
                train_corr = pearsonr(data.train_y[train_idx], data.train_c[train_idx])[0]
                corr_diff = round(train_corr - test_corr, 1)
                if ti == 0:
                    corr_diffs.append(corr_diff)
                    print('train_bias=', train_bias, 'train_corr=', train_corr,
                          'test_bias=', test_bias, 'test_corr=', test_corr,
                          'corr_diff=', corr_diff)
                    
                # Train and test each model.
                for name, model in models:
                    clf = model(data.train_x[train_idx], data.train_y[train_idx],
                                data.train_c[train_idx], rand, data.feature_names)
                    if name.find('backdoor') != -1:
                        print('model=%s, coef[\'horror\']=%f' % (name, clf.coef_[0,idx_horror][0]))
                    y_pred = clf.predict(data.test_x[test_idx])
                    y_true = data.test_y[test_idx]
                    for y in range(3):
                        for c in range(3):
                            k = 3*y+c
                            cond = lambda x:(c == 2 or data.test_c[x] == c) and (y == 2 or data.test_y[x] == y)
                            yc_test_idx = [i for i, j in enumerate(test_idx) if cond(j)]
                            accuracies[k][name].append({'test_bias': test_bias, 'train_bias': train_bias,
                                                        'corr_diff': corr_diff,
                                                        'acc': accuracy_score(y_true[yc_test_idx],
                                                                              y_pred[yc_test_idx])})
                        
    return accuracies, corr_diffs, test_biases

def sample_from_hist(data, hist, bin_edges, size):
    probas = hist/np.sum(hist)
    data_hist, data_bin_edges = np.histogram(data, bins=len(bin_edges)-1)
    diff_bins = [bisect.bisect_left(bin_edges, x)-1 for x in data]
    diff_proba = [probas[x]/data_hist[x] for x in diff_bins]
    diff_proba /= np.sum(diff_proba)
    choices = np.random.choice(range(data.shape[0]), size=size, replace=False, p=diff_proba)
    return choices

def make_confounding_data_from_hist(X, y, c, d, hist, bin_edges, pos_prob, bias, size, rand, plotting=False):
    both_pos = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 1])
    both_neg = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 0])
    ypos_cneg = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 0])
    yneg_cpos = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 1])
    
    # if bias=.9, then 90% of instances where c=1 will also have y=1
    # similarly, 10% of instances where c=1 will have y=0
    cprob = 1. * sum(c) / len(c)
    n_cpos = int(cprob * size)
    n_cneg = size - n_cpos
    n_ypos = int(pos_prob * size)
    n_yneg = size - n_ypos
    n_11 = int(bias * n_cpos)
    n_01 = int((1 - bias) * n_cpos)
    n_10 = n_ypos - n_11
    n_00 = n_yneg - n_01  
    
    s_11 = sample_from_hist(d[both_pos], hist, bin_edges, n_11)
    s_00 = sample_from_hist(d[both_neg], hist, bin_edges, n_00)
    s_10 = sample_from_hist(d[ypos_cneg], hist, bin_edges, n_10)
    s_01 = sample_from_hist(d[yneg_cpos], hist, bin_edges, n_01)
    if plotting:
        diff_plot = {0: d[both_neg[s_00]], 1: d[yneg_cpos[s_01]], 2: d[ypos_cneg[s_10]], 3: d[both_pos[s_11]]}
#         print(diff_plot)
        plot_difficulties(diff_plot, bins=20)
    
    r = np.hstack([both_pos[s_11], both_neg[s_00], ypos_cneg[s_10], yneg_cpos[s_01]])
    return r


def do_confounding_trials_from_hist(models, data, ntrials, rand):  
    test_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    train_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    corr_diffs = []
    accuracies = defaultdict(lambda: defaultdict(lambda: []))
    idx_horror = np.where(data.feature_names == 'horror')
    for train_bias in train_biases:
        for test_bias in test_biases:
            for ti in range(ntrials):
                # Sample training and testing indices.
                test_idx = make_confounding_data_from_hist(X=data.test_x, y=data.test_y, c=data.test_c, d=data.test_d,
                                                           hist=data.hist, bin_edges=data.bin_edges, pos_prob=.5,
                                                           bias=test_bias, size=1500, rand=rand)  
                test_corr = pearsonr(data.test_y[test_idx], data.test_c[test_idx])[0]
                train_idx = make_confounding_data_from_hist(X=data.train_x, y=data.train_y, c=data.train_c, d=data.train_d,
                                                           hist=data.hist, bin_edges=data.bin_edges, pos_prob=.5,
                                                           bias=train_bias, size=1500, rand=rand)    
                train_corr = pearsonr(data.train_y[train_idx], data.train_c[train_idx])[0]
                corr_diff = round(train_corr - test_corr, 1)
                if ti == 0:
                    corr_diffs.append(corr_diff)
                    print('train_bias=', train_bias, 'train_corr=', train_corr,
                          'test_bias=', test_bias, 'test_corr=', test_corr,
                          'corr_diff=', corr_diff)
                    
                # Train and test each model.
                for name, model in models:
                    clf = model(data.train_x[train_idx], data.train_y[train_idx],
                                data.train_c[train_idx], rand, data.feature_names)
                    if name.find('backdoor') != -1:
                        print('model=%s, coef[\'horror\']=%f' % (name, clf.coef_[0,idx_horror][0]))
                    y_pred = clf.predict(data.test_x[test_idx])
                    y_true = data.test_y[test_idx]
                    for y in range(3):
                        for c in range(3):
                            k = 3*y+c
                            cond = lambda x: (c == 2 or data.test_c[x] == c) and (y == 2 or data.test_y[x] == y)
                            yc_test_idx = [i for i, j in enumerate(test_idx) if cond(j)]
                            accuracies[k][name].append({'test_bias': test_bias, 'train_bias': train_bias,
                                                        'corr_diff': corr_diff,
                                                        'acc': accuracy_score(y_true[yc_test_idx],
                                                                              y_pred[yc_test_idx])})
                        
    return accuracies, corr_diffs, test_biases

