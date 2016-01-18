def find_confounder(X, y, feature_names, rand):
    """ Find a confounding feature, defined as one that appears frequently but has low chi2 value.
    X ..............Binary feature matrix
    y ..............true labels
    feature_names...list of feature names
    rand............Random state
    
    Returns:
    confounding_feature_index....index of the confounding feature
    best_feature_index...........index of feature with strongest chi2 (for comparison)
    labels.......................binary labels for the confounding feature (i.e., 1 if instance has term, 0 otherwise)
    """
    chi, p = chi2(X, y)
    counts = X.sum(0).tolist()[0]
    picked = -1
    l_thresh = len(y) * .30  # confounder must occur in at least 30% of instances.
    u_thresh = len(y) * .40  # confounder must occur in at most 40% of instances.
    for i in np.argsort(chi):
        if counts[i] > l_thresh and counts[i] < u_thresh:
            picked = i
            break
    c = np.array([1 if X[row, picked] else 0 for row in range(len(y))])
    print('picked confounding feature=', feature_names[picked], 'index=', picked, 'corr=', pearsonr(y, c)[0], 'chi=', chi[i], 'count=', counts[i])
    return picked, np.array([1 if X[row, picked] else 0 for row in range(len(y))])

def find_confounder_by_cluster(X, y, feature_names, rand):
    """ Find a confounding feature, defined as ...
    X ..............Binary feature matrix
    y ..............true labels
    feature_names...list of feature names
    rand............Random state
    
    Returns:
    confounding_feature_index....index of the confounding feature
    best_feature_index...........index of feature with strongest chi2 (for comparison)
    labels.......................binary labels for the confounding feature (i.e., 1 if instance has term, 0 otherwise)
    """
    n_clust = 10
    clusterer = KMeans(n_clusters=n_clust, init='random', n_init=10, random_state=rand)
    clusters = clusterer.fit_predict(TfidfTransformer().fit_transform(X))
    for cid in range(n_clust):
        labels = [1 if c==cid else 0 for c in clusters]
        corr = pearsonr(labels, y)[0]
        print('corr=', corr, 'size=', sum(labels))
        if abs(corr) < .3 and sum(labels) > .1 * len(y) and sum(labels) < .5 * len(y):
            print('picking cluster %d size=%d corr=%g' % (cid, sum(labels), corr))
            chi, F = chi2(X, labels)
            top_feats = np.argsort(chi)
            print('top features=%s' % (feature_names[top_feats[-10:]]))
            return top_feats[-1], np.array(labels)
    print('cant find good cluster')
