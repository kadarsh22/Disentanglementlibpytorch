import torch
import numpy as np
import logging

TINY = 1e-12


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        truncated_normal_(m.weight.data, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def truncated_normal_(tensor, mean=0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class log_gaussian:

    def __call__(self, x, mu):
        var = torch.ones(mu.shape[0], mu.shape[1]).cuda()
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logging.info(net)
    logging.info('Total number of parameters: %d' % num_params)


def mse(predicted, target):
    """mean square error """
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted  # (n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target  # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0]  # value not array


def rmse(predicted, target):
    """ root mean square error """
    return np.sqrt(mse(predicted, target))


def nmse(predicted, target):
    """ normalized mean square error """
    return mse(predicted, target) / np.var(target)


def nrmse(predicted, target):
    """ normalized root mean square error """
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
    """p: probabilities """
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))


def entropic_scores(r):
    """r: relative importances"""
    r = np.abs(r)
    ps = r / np.sum(r, axis=0)  # 'probabilities'
    hs = [1 - norm_entropy(p) for p in ps.T]
    return hs
