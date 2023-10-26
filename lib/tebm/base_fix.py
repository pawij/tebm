# Fixed time interval Temporal Event-Based Model
# Base class
# Author: Peter Wijeratne (p.wijeratne@sussex.ac.uk)

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator

from . import _tebm_fix

import warnings
warnings.filterwarnings('ignore', message='divide by zero encountered in log')

class BaseTEBM(BaseEstimator):

    def __init__(self,
                 X=None,
                 lengths=None,
                 n_stages=None,
                 time_mean=None,
                 n_iter=None,
                 fwd_only=False,
                 order=None,
                 algo='viterbi',
                 verbose=False):
        self.X = X
        self.lengths = lengths
        self.n_stages = n_stages
        if time_mean:
            self.time_mean = time_mean
        else:
            self.time_mean = 1
        self.n_iter = n_iter
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        self.fwd_only = fwd_only
        if order:
            self.order = order
        else:
            self.order = self.n_stages-1
        self.algo = algo
        # currently only do a single EM iteration, but this might change in the future
        self.tol = 1E-3
        self.verbose = verbose
        # initialise p_vec and a_mat
        self.p_vec_prior = np.full(self.n_stages, 1./self.n_stages)
        self.p_vec = self.p_vec_prior
        self.a_mat_prior = np.ones((n_stages,n_stages))
        if self.fwd_only:
            for i in range(len(self.a_mat_prior)):
                self.a_mat_prior[i,i] = self.time_mean
                self.a_mat_prior[i,:i] = 0.
                if (i+self.order+1) < len(self.a_mat_prior):
                    self.a_mat_prior[i,i+self.order+1:] = 0.
                count_nonzero = np.count_nonzero(self.a_mat_prior[i]!=0)
                # distribute probability to nonzero states
                for j in range(n_stages):
                    #                    self.a_mat_prior[i,:i] = 0.
                    if i!=j and self.a_mat_prior[i,j]!=0.:
                        self.a_mat_prior[i,j] = (1-self.a_mat_prior[i,i])/(count_nonzero-1)
                    elif i==(n_stages-1) and (j==n_stages-1):
                        self.a_mat_prior[i,j] = 1.
        else:
            self.a_mat_prior = np.full((self.n_stages, self.n_stages), 1./self.n_stages)
        self.a_mat = self.a_mat_prior

    def reinit(self):
        self.p_vec = self.p_vec_prior
        self.a_mat = self.a_mat_prior

    def compute_forward(self, loglike_i):
        n_samples, n_stages = loglike_i.shape
        alpha_i = np.zeros((n_samples, n_stages))
        _tebm_fix._forward(n_samples,
                           n_stages,
                           np.log(self.p_vec),
                           np.log(self.a_mat),
                           loglike_i,
                           alpha_i)
        return alpha_i

    def compute_backward(self, loglike_i):
        n_samples, n_stages = loglike_i.shape
        beta_i = np.zeros((n_samples, n_stages))
        _tebm_fix._backward(n_samples,
                            n_stages,
                            np.log(self.p_vec),
                            np.log(self.a_mat),
                            loglike_i,
                            beta_i)
        return beta_i

    def compute_posteriors(self, alpha_i, beta_i):
        post = alpha_i + beta_i
        post -= logsumexp(post, axis=1, keepdims=True)
        return np.exp(post)

    def update_params(self,
                      p_vec,
                      a_mat,
                      loglike_i,
                      post_i,
                      alpha_i,
                      beta_i):
        # initial probability
        p_vec += post_i[0]
        # transition matrix
        n_samples, n_stages = loglike_i.shape
        # skip if only one observation - no temporal info
        if n_samples == 1:
            return
        log_prob_tau = np.full((n_stages, n_stages), -np.inf)
        _tebm_fix._compute_log_prob_tau(n_samples,
                                        n_stages,
                                        alpha_i,
                                        np.log(self.a_mat),
                                        beta_i,
                                        loglike_i,
                                        log_prob_tau)         
        a_mat += np.exp(log_prob_tau)

    def m_step(self, p_vec, a_mat):
        # update initial probability
        # apply prior
        p_vec = np.maximum(self.p_vec_prior - 1 + p_vec, 0)
        # prevent forbidden transitions
        self.p_vec = np.where(self.p_vec == 0, 0, p_vec)
        # normalise
        self.p_vec = self.p_vec / self.p_vec.sum()
        # update transition matrix
        # apply prior
        a_mat = np.maximum(self.a_mat_prior - 1 + a_mat, 0)
        # prevent forbidden transitions
        self.a_mat = np.where(self.a_mat == 0, 0, a_mat)
        # normalise
        row_sums = self.a_mat.sum(axis=1)
        row_sums[row_sums==0] = 1
        self.a_mat = self.a_mat / row_sums[:, np.newaxis]

    def fit(self):
        self.reinit()
        curr_loglike = -np.inf
        for n in range(self.n_iter):
            p_vec = np.zeros(self.n_stages)
            a_mat = np.zeros((self.n_stages, self.n_stages))
            loglike = 0
            for i in range(len(self.lengths)):
                s_idx, e_idx = int(np.sum(self.lengths[:i])), int(np.sum(self.lengths[:i])+self.lengths[i])
                X_i = self.X[s_idx:e_idx]
                loglike_i = self.compute_log_likelihood(X_i, s_idx, e_idx)
                alpha_i = self.compute_forward(loglike_i)
                beta_i = self.compute_backward(loglike_i)
                post_i = self.compute_posteriors(alpha_i, beta_i)
                self.update_params(p_vec, a_mat, loglike_i, post_i, alpha_i, beta_i)
                loglike += logsumexp(alpha_i[-1])
            self.m_step(p_vec, a_mat)
            # check likelihood for convergence - currently we don't use this, as default self.n_iter = 1
            if self.verbose:
                print (n, loglike-curr_loglike)
            if loglike-curr_loglike < self.tol:
                break
            curr_loglike = loglike

    def compute_viterbi(self, X, i, j):
        loglike_i = self.compute_log_likelihood(X, i, j)
        n_samples, n_stages = loglike_i.shape
        stages = _tebm_fix._viterbi(n_samples,
                                    n_stages,
                                    np.log(self.p_vec),
                                    np.log(self.a_mat),
                                    loglike_i)
        return stages

    def compute_map(self, X, i, j):
        posteriors = self.posteriors(X)        
        stages = np.argmax(posteriors, axis=1)
        return stages

    def posteriors_X(self, X, lengths=None):
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.n_stages))
        for i in range(len(lengths)):
            s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+lengths[i])
            X_i = X[s_idx:e_idx]
            loglike_i = self.compute_log_likelihood(X_i, s_idx, e_idx)
            alpha_i = self.compute_forward(loglike_i)
            beta_i = self.compute_backward(loglike_i)
            posteriors[s_idx:e_idx] = self.compute_posteriors(alpha_i, beta_i)
        return posteriors

    def prob_X(self, X, lengths=None):
        n_samples = X.shape[0]
        prob = np.zeros((n_samples, self.n_stages))
        for i in range(len(lengths)):
            s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+lengths[i])
            X_i = X[s_idx:e_idx]
            loglike_i = self.compute_log_likelihood(X_i, s_idx, e_idx)
            alpha_i = self.compute_forward(loglike_i)
            prob[s_idx:e_idx] = alpha_i
        return prob

    def compute_model_log_likelihood(self, X, lengths=None):
        loglike = 0
        for i in range(len(lengths)):
            s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+lengths[i])
            X_i = X[s_idx:e_idx]
            loglike_i = self.compute_log_likelihood(X_i, s_idx, e_idx)
            alpha_i = self.compute_forward(loglike_i)
            loglike += logsumexp(alpha_i[-1])
        return loglike

    def stage_X(self, X, lengths=None, algo=None):
        if self.algo == 'viterbi':
            stage_algo = self.compute_viterbi
        elif self.algo == 'map':
            stage_algo = self.compute_map        
        n_samples = X.shape[0]
        stages = np.empty(n_samples, dtype=int)
        for i in range(len(lengths)):
            s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+lengths[i])
            X_i = X[s_idx:e_idx]
            stagesij = stage_algo(X_i, s_idx, e_idx)
            stages[s_idx:e_idx] = stagesij
        return stages
