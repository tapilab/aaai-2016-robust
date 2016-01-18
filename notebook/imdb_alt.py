def compute_difficulties(X, y, c, rand=np.random.RandomState(1234567),
                         feature_names=None, train_ratio=.5, model=lr):
    difficulties = defaultdict(list)
    cur_train = int(train_ratio*X.shape[0])
    indices = np.arange(X.shape[0])
    rand.shuffle(indices)
    X_shuf, y_shuf, c_shuf = (X[indices], y[indices], c[indices])
    X_tr, y_tr, c_tr = X_shuf[:cur_train], y_shuf[:cur_train], c_shuf[:cur_train]
    X_te, y_te, c_te = X_shuf[cur_train:], y_shuf[cur_train:], c_shuf[cur_train:]
    clf = model(X_tr, y_tr, c_tr, rand, feature_names)
    pred_prob = clf.predict_proba(X_te)
    for y_true, y_pred_prob, c_val in zip(y_te, pred_prob, c_te):
        difficulties[2*y_true+c_val].append(1-y_pred_prob[y_true])
    return difficulties

def search_coef_diff(data, ncoef=20):
    """Fit a LR classifier on the training data and a LR classifier on the testing data and print the n features with the
    highest shift in coefficient"""
    rand = np.random.RandomState(1234567)
    clf_tr = lr(data.train_x, data.train_y, data.train_c, rand, data.feature_names)
    clf_te = lr(data.test_x, data.test_y, data.train_c, rand, data.feature_names)
    tr_coef = clf_tr.coef_[0]
    te_coef = clf_te.coef_[0]
    diff_coef = np.abs(tr_coef - te_coef)
    desc_diff_coef_idx = np.argsort(diff_coef)[::-1]
    top_n_coef = diff_coef[desc_diff_coef_idx[:ncoef]]
    line = lambda c_idx:'\tname=%s\tte_coef=%f\ttr_coef=%f\tdiff_coef=%f' % (data.feature_names[c_idx],
                                                                             te_coef[c_idx],
                                                                             tr_coef[c_idx],
                                                                             diff_coef[c_idx])
    print ('\n'.join([line(c_idx) for c_idx in desc_diff_coef_idx[:ncoef]]))
    
def search_coef_diff_train_test(data, ntrials, ncoef=20, model=lr, remove_last_features=False):
    """Fit a LR classifier on the training set and a LR classifier on the testing set for every possible bias values.
    Then print the features with the highest shift in coefficient. Also plots a matrix of the average n highest shift
    for every testing bias/training bias pair."""
    rand = np.random.RandomState(1234567)
    
    test_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    train_biases = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    
    all_diff_coef = np.zeros((len(test_biases),len(train_biases)))
    for te, test_bias in enumerate(test_biases):
        for tr, train_bias in enumerate(train_biases):
            tr_coefs = []
            te_coefs = []
            for ti in range(ntrials):
                # Sample training and testing indices.
                test_idx = make_confounding_data(X=data.test_x, y=data.test_y, c=data.test_c,
                                                pos_prob=.5, bias=test_bias, size=6000, rand=rand) 
                train_idx = make_confounding_data(X=data.train_x, y=data.train_y, c=data.train_c,
                                                  pos_prob=.5, bias=train_bias, size=6000, rand=rand)

                clf_tr = model(data.train_x[train_idx], data.train_y[train_idx],
                                data.train_c[train_idx], rand, data.feature_names)
                clf_te = model(data.test_x[test_idx], data.test_y[test_idx],
                                data.test_c[test_idx], rand, data.feature_names)
                if ti == 0:
                    print('- test_bias=', test_bias, 'train_bias=', train_bias)
                tr_coef = clf_tr.coef_[0]
                te_coef = clf_te.coef_[0]
                if remove_last_features:
                    tr_coef = clf_tr.coef_[0][:-remove_last_features]
                    te_coef = clf_te.coef_[0][:-remove_last_features]
                tr_coefs.append(tr_coef)
                te_coefs.append(te_coef)
            
            avg_tr_coef = np.mean(tr_coefs, axis=0)
            avg_te_coef = np.mean(te_coefs, axis=0)
            diff_coef = np.abs(avg_tr_coef - avg_te_coef)
            
            desc_diff_coef_idx = np.argsort(diff_coef)[::-1]
            top_n_coef = diff_coef[desc_diff_coef_idx[:ncoef]]
            all_diff_coef[te,tr] = np.mean(top_n_coef)
            line = lambda c_idx:'\tname=%s\tte_coef=%f\ttr_coef=%f\tdiff_coef=%f' % (data.feature_names[c_idx],
                                                                                     clf_te.coef_[0][c_idx],
                                                                                     clf_tr.coef_[0][c_idx],
                                                                                     diff_coef[c_idx])
            print ('\n'.join([line(c_idx) for c_idx in desc_diff_coef_idx[:ncoef]]))
    print(all_diff_coef)
    plot_matrix(all_diff_coef,
                'Avg coefficient difference for the %d most different features' % ncoef,
                train_biases, test_biases)
