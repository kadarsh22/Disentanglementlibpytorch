import torch
import numpy as np

TINY = 1e-12

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)

def normalize(X, mean=None, stddev=None, useful_features=None, remove_constant=True):
    calc_mean, calc_stddev = False, False

    if mean is None:
        mean = np.mean(X, 0)  # training set
        calc_mean = True

    if stddev is None:
        stddev = np.std(X, 0)  # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0]  # inconstant features, ([0]=shape correction)

    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]

    X_zm = X - mean
    X_zm_unit = X_zm / stddev

    return X_zm_unit, mean, stddev, useful_features


def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))


def entropic_scores(r):
    '''r: relative importances '''
    r = np.abs(r)
    ps = r / np.sum(r, axis=0)  # 'probabilities'
    hs = [1 - norm_entropy(p) for p in ps.T]
    return hs