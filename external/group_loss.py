"""
MIT License

Copyright (c) 2020  Dynamic Vision and Learning Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U

def _replicator(W, X, tol, max_iter):
    """
    Replicator Dynamics
    Output:
    X:  the population(s) at convergence
    i:  the number of iterations needed to converge
    prec:  the precision reached by the dynamical system
    """

    i = 0
    while i < max_iter:
        X = X * torch.matmul(W, X)
        X /= X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1)
        i += 1

    return X

def dynamics(W, X, tol=1e-6, max_iter=5, mode='replicator', **kwargs):
    """
    Selector for dynamics
    Input:
    W:  the pairwise nxn similarity matrix (with zero diagonal)
    X:  an (n,m)-array whose rows reside in the n-dimensional simplex
    tol:  error tolerance
    max_iter:  maximum number of iterations
    mode: 'replicator' to run the replicator dynamics
    """

    if mode == 'replicator':
        X = _replicator(W, X, tol, max_iter)
    else:
        raise ValueError('mode \'' + mode + '\' is not defined.')

    return X


class GTG(nn.Module):
    def __init__(self, total_classes, tol=-1., max_iter=5, mode='replicator', device='cuda:0'):
        super(GTG, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.device = device

    def _init_probs_uniform(self, labs, L, U):
        """ Initialized the probabilities of GTG from uniform distribution """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        ps[L, labs] = 1.
        
        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            might need tuning in order to reach the same performance as _init_probs_prior """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        ps[L, labs] = 1.
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def _get_W(self, x):

        x = (x - x.mean(dim=1).unsqueeze(1))
        norms = x.norm(dim=1)
        W = torch.mm(x, x.t()) / torch.ger(norms, norms)

        W = self.set_negative_to_zero(W.cuda())
        return W

    def forward(self, fc7, num_points, labs, L, U, probs=None, classes_to_use=None):
        W = self._get_W(fc7)
        if type(probs) is type(None):
            ps = self._init_probs_uniform(labs, L, U)
        else:
            if type(classes_to_use) is type(None):
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
            else:
                ps = probs
                ps = self._init_probs_prior_only_classes(ps, labs, L, U, classes_to_use)
        ps = dynamics(W, ps, self.tol, self.max_iter, self.mode)
        return ps, W
