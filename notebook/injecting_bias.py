import numpy as np

def make_confounding_data(X, y, c, pos_prob, bias, size, rand=np.random.RandomState(123456)):
    """ Create Sample a dataset of given size where c is a confounder for y with strength=bias.
        We take care not to introduce selection bias (that is, p(c=1) is representative of training data).
        This assumes that #[c=1] < #[y=1].
        
        X: data matrix
        y: labels (0, 1)
        c: confounding labels (0,1)
        pos_prop: proportion of instances where y=1
        bias: amount of bias (0-1)
        size: number of samples
        rand: RandomState
    """
    both_pos = [i for i in range(len(y)) if y[i] == 1 and c[i] == 1]
    both_neg = [i for i in range(len(y)) if y[i] == 0 and c[i] == 0]
    ypos_cneg = [i for i in range(len(y)) if y[i] == 1 and c[i] == 0]
    yneg_cpos = [i for i in range(len(y)) if y[i] == 0 and c[i] == 1]
    
    for x in [both_pos, both_neg, yneg_cpos, ypos_cneg]:
#         print(len(x))
        rand.shuffle(x)

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
    
    r = np.array(both_pos[:n_11] + both_neg[:n_00] + ypos_cneg[:n_10] + yneg_cpos[:n_01])
    return r

def make_confounding_data_from_subsampling(X, y, c, size, rand=np.random.RandomState(111191)):
    """
    Sample size/4 instances from each for each of the possible c/y combinations.
    """
    both_pos = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 1])
    both_neg = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 0])
    ypos_cneg = np.array([i for i in range(len(y)) if y[i] == 1 and c[i] == 0])
    yneg_cpos = np.array([i for i in range(len(y)) if y[i] == 0 and c[i] == 1])
    
    lengths = []
    samples = []
#     diffs = []
    l = 1.*size/4.
    for x in [both_pos, both_neg, yneg_cpos, ypos_cneg]:
        rand.shuffle(x)
        s = x[:int(l)]
        samples.append(s)
#         diffs.append(d[s])
#     plt.hist(diffs)
    r = np.hstack(samples)
    return r
