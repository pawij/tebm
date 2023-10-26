# Variable time interval Temporal Event-Based Model
# Derived class from base_var.py
# Author: Peter Wijeratne (p.wijeratne@sussex.ac.uk)

import logging

import numpy as np
import scipy as sp
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state

from . import _utils
from .stats import log_multivariate_normal_density
from .base_var import _BaseTEBM
from .utils import (
    fill_covars, iter_from_X_lengths, log_mask_zero, log_normalize, normalize)
# change this for MixtureTEBM and ZscoreTEBM (and...)
__all__ = ["MixtureTEBM", "ZscoreTEBM", "MultinomialTEBM", "GMMTEBM"]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))

#PW
import multiprocessing
from functools import partial
import pathos
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models, get_prob_mat

def _check_and_set_gaussian_n_features(model):
    _, n_features = model.X.shape
    if hasattr(model, "n_features") and model.n_features != n_features:
        raise ValueError("Unexpected number of dimensions, got {} but "
                         "expected {}".format(n_features, model.n_features))
    model.n_features = n_features


class MixtureTEBM(_BaseTEBM):
    r"""Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"
            (n_features, n_features)                if "tied",

    Examples
    --------
    >>> from tebm.tebm import MixtureTEBM
    >>> MixtureTEBM(n_components=2)  #doctest: +ELLIPSIS
    MixtureTEBM(algorithm='viterbi',...
    """
    def __init__(self, X=None, lengths=None, jumps=None, 
                 n_components=1, time_mean=None, startprob_prior=None, transmat_prior=None,
                 means_prior=0, means_weight=0, covars_prior=1e-2, covars_weight=1, covariance_type='diag', min_covar=1e-3,
                 algorithm="viterbi", random_state=None, n_iter=10,
                 tol=1e-2, verbose=False, params="st",
                 init_params="st", allow_nan=False,
                 fwd_only=False, order=None):
        _BaseTEBM.__init__(self, X=X, lengths=lengths, jumps=jumps, 
                           n_components=n_components, time_mean=time_mean, startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                           algorithm=algorithm, random_state=random_state, n_iter=n_iter,
                           tol=tol, verbose=verbose, params=params,
                           init_params=init_params, allow_nan=allow_nan,
                           fwd_only=fwd_only, order=order)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        _utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars
    """
    def _check(self):
        super()._check()
        
        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {}'
                             .format(COVARIANCE_TYPES))
    """
    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1)
        }

    def _init(self, X=None, lengths=None, jumps=None):
        _check_and_set_gaussian_n_features(self)
        self.startprob_ = self.startprob_prior
        self.Q_ = self.Q_prior
        #FIXME: check
        if len(self.step_sizes):
            self.transmat_ = np.array([self.transmat_prior[i] for i in range(len(self.step_sizes))])
        else:
            self.transmat_ = np.array([self.transmat_prior])
        #        self.transmat_ = np.array([sp.linalg.expm(self.step_sizes[i]*self.Q_prior) for i in range(len(self.step_sizes))])

    def gen_sample(self, n_samples=1, scale=1):
        ### FIXME: this won't work for KDE EBM
        def _get_params_ebm(means, sdevs, mixes):
            n_bms = self.n_features
            seq_means = np.tile(means.T[0], (n_bms+1, 1)).T
            seq_sdevs = np.tile(sdevs.T[0], (n_bms+1, 1)).T
            seq_mixes = np.tile(mixes.T[0], (n_bms+1, 1)).T
            # seq_means[0] = healthy distributions for all biomarkers
            for i in range(n_bms):
                bm_pos = np.where(self.S == i)[0][0]
                seq_means[i, bm_pos+1:] = means[i][1]
                seq_sdevs[i, bm_pos+1:] = sdevs[i][1]
                seq_mixes[i, bm_pos+1:] = mixes[i][1]
            return seq_means.T, seq_sdevs.T, seq_mixes.T
        def _return_gmm_fits(mixtures):
            n_bms = self.n_features
            fit_means = np.zeros((n_bms, 2))
            fit_std = np.zeros((n_bms, 2))
            fit_mixes = np.zeros((n_bms, 2))
            for i in range(n_bms):
                theta_i = mixtures[i].theta
                fit_means[i] = theta_i[[0,2]]
                fit_std[i] = theta_i[[1,3]]
                fit_mixes[i] = [theta_i[4],1-theta_i[4]]
            return fit_means, fit_std, fit_mixes
        fit_means, fit_std, fit_mixes = _return_gmm_fits(self.mixtures)
        theta = _get_params_ebm(fit_means, fit_std, fit_mixes)
        self.means = theta[0]
        self.covars = np.power(theta[1],2)
        # generate samples
        X_sample, k_sample, dt_sample = [], [], []
        # sample initial state from model
        k_i = np.random.choice(self.n_components, size=1, p=self.startprob_)[0]
        # sample initial observation from model with corresponding stage
        X_sample.append(np.random.normal(self.means[k_i], np.sqrt(self.covars[k_i])))
        k_sample.append(k_i)
        # initial observation time is always zero
        dt_sample.append(0)
        for i in range(1,n_samples):
            # sample jump from exponential distribution with scale factor >= 1
            dt_i = int(np.random.exponential(scale))+1
            dt_sample.append(dt_i)
            # sample transition probability from model with corresponding jump
            weights = np.real(sp.linalg.expm(self.Q_*dt_i))[k_i,:].flatten()
            k_i = np.random.choice(self.n_components, size=1, p=weights)[0]
            k_sample.append(k_i)
            # sample observation from model with corresponding stage
            # FIXME: make multivariate            
            X_sample.append(np.random.normal(self.means[k_i], np.sqrt(self.covars[k_i])))
        # sample a single diagnosis label
        #FIXME: how best to generate a label?
        CASE_STAGE_THRESHOLD = 0
        label_sample = 1 if k_sample[0] > CASE_STAGE_THRESHOLD else 0
        return np.array(X_sample), np.array(k_sample), np.array(dt_sample), label_sample

    def _compute_log_likelihood(self, X, start_i, end_i):
        n_samples = end_i-start_i
        S_int = self.S.astype(int)
        arange_Np1 = np.arange(0, self.n_features+1)
        p_perm_k = np.zeros((n_samples, self.n_features+1))
        p_yes = np.array(self.prob_mat[start_i:end_i, :, 1])
        p_no = np.array(self.prob_mat[start_i:end_i, :, 0])
        # Leon's clever cumulative probability code
        cp_yes = np.cumprod(p_yes[:, S_int], 1)
        cp_no = np.cumprod(p_no[:, S_int[::-1]], 1)
        for i in arange_Np1:
            if i == 0:
                p_perm_k[:, i] = cp_no[:,self.n_features-1]
            elif i == self.n_features:
                p_perm_k[:, i] = cp_yes[:,self.n_features-1]
            else:
                p_perm_k[:, i] = cp_yes[:,i-1] * cp_no[:,self.n_features-i-1]
        p_perm_k[p_perm_k==0] = np.finfo(float).eps
        return np.log(p_perm_k)

    def score_samples(self, X, lengths=None, jumps=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        self.prob_mat = get_prob_mat(X, self.mixtures)
        ###                
        return super().score_samples(X, lengths, jumps)
        
    def predict(self, X, lengths=None, jumps=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        self.prob_mat = get_prob_mat(X, self.mixtures)
        ###
        logprob, state_sequence = self.decode(X, lengths, jumps)
        return state_sequence, logprob

    def predict_proba(self, X, lengths=None, jumps=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        self.prob_mat = get_prob_mat(X, self.mixtures)
        ###
        _, posteriors = self.score_samples(X, lengths, jumps)
        return posteriors
    
    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        ### FIXME: this won't work for KDE EBM
        def _get_params_ebm(means, sdevs, mixes):
            n_bms = self.n_features
            seq_means = np.tile(means.T[0], (n_bms+1, 1)).T
            seq_sdevs = np.tile(sdevs.T[0], (n_bms+1, 1)).T
            seq_mixes = np.tile(mixes.T[0], (n_bms+1, 1)).T
            # seq_means[0] = healthy distributions for all biomarkers
            for i in range(n_bms):
                bm_pos = np.where(self.S == i)[0][0]
                seq_means[i, bm_pos+1:] = means[i][1]
                seq_sdevs[i, bm_pos+1:] = sdevs[i][1]
                seq_mixes[i, bm_pos+1:] = mixes[i][1]
            return seq_means.T, seq_sdevs.T, seq_mixes.T
        def _return_gmm_fits(mixtures):
            n_bms = self.n_features
            fit_means = np.zeros((n_bms, 2))
            fit_std = np.zeros((n_bms, 2))
            fit_mixes = np.zeros((n_bms, 2))
            for i in range(n_bms):
                theta_i = mixtures[i].theta
                fit_means[i] = theta_i[[0,2]]
                fit_std[i] = theta_i[[1,3]]
                fit_mixes[i] = [theta_i[4],1-theta_i[4]]
            return fit_means, fit_std, fit_mixes
        fit_means, fit_std, fit_mixes = _return_gmm_fits(self.mixtures)
        theta = _get_params_ebm(fit_means, fit_std, fit_mixes)
        self.means_ = theta[0]
        self.covars_ = np.power(theta[1],2)
        ###
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors,
                                          fwdlattice, bwdlattice, framejumps):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice, framejumps)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

    def _optimise_seq(self, S):
        N = self.n_features
        max_S = S.copy()
        seen = []
        # calculate likelihoods over permutations
        order_bio = np.random.permutation(N)        
        for count,i in enumerate(order_bio):
            current_sequence = max_S
            assert(len(current_sequence)==N)
            current_location = np.array([0] * N)
            current_location[current_sequence.astype(int)] = np.arange(N)
            selected_event = i
            move_event_from = current_location[selected_event]
            possible_positions = np.arange(N)
            possible_sequences = np.zeros((len(possible_positions), N))
            possible_likelihood = np.full((len(possible_positions), 1), -np.inf)
            for index in range(len(possible_positions)):
                current_sequence = max_S
                # choose a position in the sequence to move an event to
                move_event_to = possible_positions[index]
                # move this event in its new position
                current_sequence = np.delete(current_sequence, move_event_from, 0)
                new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                # skip if already calculated likelihood for this sequence
                if list(new_sequence) in seen:
                    continue
                seen.append(list(new_sequence))                
                # fit TEBM
                self.S = new_sequence
                self.fit()
                possible_likelihood[index] = self.score(self.X, self.lengths, self.jumps)
                possible_sequences[index, :] = self.S
                if np.all(new_sequence==np.array([[9,5,3,6,4,0,2,8,7,1]])) or np.all(new_sequence==np.array([[9,5,3,6,4,0,8,7,2,1]])):
                    print (new_sequence,possible_likelihood[index])
            max_likelihood = max(possible_likelihood)
            max_S = possible_sequences[np.where(possible_likelihood == max_likelihood)[0][0]]
            if count<(N-1):
                print (str(round((count+1)/len(order_bio)*100,2))+'% complete')
        return max_S, max_likelihood

    def _seq_em(self, S, n_iter, seed_num):
        # parse out sequences by seed number
        S = np.array(S[seed_num])
        print ('Startpoint',seed_num)
        cur_seq = S
        cur_like = -np.inf
        flag = False
        for opt_i in range(int(n_iter)):
            print ('EM iteration',opt_i+1)
            seq, like = self._optimise_seq(cur_seq)
            print ('current', like, seq, 'max', cur_like, cur_seq)
            if like-cur_like < 1E-3:
                print ('EM converged in',opt_i+1,'iterations')
                flag = True
            elif like > cur_like:
                cur_seq = seq
                cur_like = like
            if flag:
                break
        return cur_seq, cur_like

    def _fit_tebm(self, labels, n_start, n_iter, n_cores, model_type='GMM', constrained=False, cut_controls=False, X_mixture=[], lengths_mixture=[], labels_mixture=[]):
        # only use baseline data to fit mixture models
        if len(X_mixture) == 0:
            X_mixture = self.X
            lengths_mixture = self.lengths
            labels_mixture = labels
        X0 = []
        for i in range(len(lengths_mixture)):
            X0.append(X_mixture[np.sum(lengths_mixture[:i])])
        X0 = np.array(X0)
        if model_type == 'KDE':
            mixtures = fit_all_kde_models(X0, labels_mixture)
        else:
            mixtures = fit_all_gmm_models(X0, labels_mixture)#, constrained=constrained)
        # might want to fit sequence without controls
        if cut_controls:
            print ('Cutting controls from sequence fit!')
            X, lengths, jumps = [], [], []
            for i in range(len(self.lengths)):
                if labels[i] != 0:
                    nobs_i = self.lengths[i]
                    for x in self.X[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        X.append(x)
                    for x in self.jumps[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        jumps.append(x)
                    lengths.append(self.lengths[i])
            self.X = np.array(X)
            self.lengths = np.array(lengths)
            self.jumps = np.array(jumps)
        # calculate likelihood lookup table
        self.prob_mat = get_prob_mat(self.X, mixtures)
        # set mixture models
        self.mixtures = mixtures
        # do EM
        ml_seq_mat = np.zeros((1,self.X.shape[1],n_start))
        ml_like_mat = np.zeros(n_start)
        if n_cores>1:
            pool = pathos.multiprocessing.ProcessingPool()
            pool.ncpus = n_cores
        else:
            # FIXME: serial version doesn't work
            #            pool = pathos.serial.SerialPool()
            pool = pathos.multiprocessing.ProcessingPool()
            pool.ncpus = n_cores
        # instantiate function as class to pass to pool.map
        # first calculate array of sequences - do this first or same random number will be used simultaneously on each processor
        # will return shape (n_start, 1)
        copier = partial(self._init_seq)
        # will return shape (n_start, 1)
        seq_mat = np.array(pool.map(copier, range(n_start)))
        # now optimise
        copier = partial(self._seq_em,
                         seq_mat[:,0],
                         n_iter)
        # will return shape (n_start, 2)
        par_mat = list(pool.map(copier, range(n_start)))
        # distribute to local matrices
        for i in range(n_start):
            ml_seq_mat[:, :, i] = par_mat[i][0]
            ml_like_mat[i] = par_mat[i][1]
        ix = np.argmax(ml_like_mat)
        ml_seq = ml_seq_mat[:, :, ix]
        ml_like = ml_like_mat[ix]
        # refit model on ML sequence
        self.S = ml_seq[0]
        self.fit()
        return ml_seq, self.mixtures

    def _init_seq(self, seed_num):
        #FIXME: issue with seeding by seed_num is that every time you call _fit_tebm, it will initialise the same sequences
        # ensure randomness across parallel processes
        np.random.seed(seed_num)
        S = np.arange(self.n_features)
        np.random.shuffle(S)
        return [S]

class ZscoreTEBM(_BaseTEBM):
    r"""Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    means\_ : array, shape (n_components, n_features)
    Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"
            (n_features, n_features)                if "tied",

    Examples
    --------
    >>> from tebm.tebm import ZscoreTEBM
    >>> ZscoreTEBM(n_components=2)  #doctest: +ELLIPSIS
    ZscoreTEBM(algorithm='viterbi',...
    """
    def __init__(self, X=None, lengths=None, jumps=None,
                 n_components=1, startprob_prior=None, transmat_prior=None,
                 means_prior=0, means_weight=0, covars_prior=1e-2, covars_weight=1, covariance_type='diag', min_covar=1e-3,
                 algorithm="viterbi", random_state=None, n_iter=10,
                 tol=1e-2, verbose=False, params="st",
                 init_params="st", allow_nan=False,
                 fwd_only=False, order=None):
        _BaseTEBM.__init__(self, X=X, lengths=lengths, jumps=jumps,
                           n_components=n_components, startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                           algorithm=algorithm, random_state=random_state, n_iter=n_iter,
                           tol=tol, verbose=verbose, params=params,
                           init_params=init_params, allow_nan=allow_nan,
                           fwd_only=fwd_only, order=order)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        # check degeneracy
        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        n_fit_scalars = sum(n_fit_scalars_per_param[p] for p in self.params)
        if self.X.size < n_fit_scalars:
            _log.warning("Fitting a model with {} free scalar parameters with "
                         "only {} data points will result in a degenerate "
                         "solution.".format(n_fit_scalars, self.X.size))

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        covars = np.array(covars, copy=True)
        _utils._validate_covars(covars, self.covariance_type,
                                self.n_components)
        self._covars_ = covars

    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {}'
                             .format(COVARIANCE_TYPES))

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _init(self, X=None, lengths=None, jumps=None):
        _check_and_set_gaussian_n_features(self)
        self.startprob_ = self.startprob_prior
        self.Q_ = self.Q_prior
        #FIXME: check
        if len(self.step_sizes):
            self.transmat_ = np.array([self.transmat_prior[i] for i in range(len(self.step_sizes))])
        else:
            self.transmat_ = np.array([self.transmat_prior])
        #        self.transmat_ = np.array([sp.linalg.expm(self.step_sizes[i]*self.Q_prior) for i in range(len(self.step_sizes))])
        if 'm' in self.init_params or not hasattr(self, "means_"):
            print ('Error! Have not set Zscore mean')
            quit()
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            print ('Error! Have not set Zscore covariance')
            quit()

    def _compute_log_likelihood(self, X, start_i, end_i):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def predict(self, X, lengths=None, jumps=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        ###
        logprob, state_sequence = self.decode(X, lengths, jumps)
        return state_sequence, logprob

    def predict_proba(self, X, lengths=None, jumps=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        ###
        _, posteriors = self.score_samples(X, lengths, jumps)
        return posteriors
    
    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.multivariate_normal(
            self.means_[state], self.covars_[state]
        )

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice, framejumps):
        super()._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice, framejumps)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))

    def _init_seq(self, seed_num):
        np.random.seed()    
        N = np.array(self.stage_zscore).shape[1]
        S = np.zeros(N)
        for i in range(N):
            IS_min_stage_zscore = np.array([False] * N)
            possible_biomarkers = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected = [False] * N
                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k] = True
                this_biomarkers = np.array([(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore = 0
                else:
                    this_min_stage_zscore = min(self.stage_zscore[this_biomarkers])
                if (this_min_stage_zscore):
                    temp = ((this_biomarkers.astype(int) + (self.stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                    temp = temp.reshape(len(temp), )
                    IS_min_stage_zscore[temp] = True
            events = np.array(range(N))
            possible_events = np.array(events[IS_min_stage_zscore])
            this_index = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
            S[i] = possible_events[int(this_index)]
        S = S.reshape(1, len(S))
        return S

    def _get_means(self):
        def linspace_local2(a, b, N, arange_N):
            return a + (b - a) / (N - 1.) * arange_N
        N = self.stage_biomarker_index.shape[1]
        S_inv = np.array([ 0 ] * N)
        S_inv[self.S.astype(int)] = np.arange(N)
        possible_biomarkers = np.unique(self.stage_biomarker_index)
        B = len(possible_biomarkers)
        # value of mean function at integral limits
        point_value = np.zeros((B, N + 2))
        # all the arange you'll need below
        arange_N = np.arange(N + 2)
        for i in range(B):
            b = possible_biomarkers[i]
            # position of this biomarker's z-score events in the sequence
            event_location = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            # z-score reached at each event
            event_value = np.concatenate([[self.min_biomarker_zscore[i]], self.stage_zscore[self.stage_biomarker_index == b], [self.max_biomarker_zscore[i]]])
            for j in range(len(event_location) - 1):
                if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    temp = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp] = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])
                else:
                    temp = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp] = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])
        # integrate (approximation)
        stage_value = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]
        return stage_value.T

    def _optimise_seq(self, S):
        N = self.stage_zscore.shape[1]
        max_S = S.copy()
        # calculate likelihoods over permutations
        order_bio = np.random.permutation(N)        
        for count,i in enumerate(order_bio):
            current_sequence = max_S
            assert(len(current_sequence)==N)
            current_location = np.array([0]*len(current_sequence))
            current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))
            selected_event = i
            move_event_from = current_location[selected_event]
            this_stage_zscore = self.stage_zscore[0, selected_event]
            selected_biomarker = self.stage_biomarker_index[0, selected_event]
            possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]
            min_filter = possible_zscores_biomarker < this_stage_zscore
            max_filter = possible_zscores_biomarker > this_stage_zscore
            events = np.array(range(N))
            if np.any(min_filter):
                min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                min_zscore_bound_event = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (
                    self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
            else:
                move_event_to_lower_bound = 0
            if np.any(max_filter):
                max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                max_zscore_bound_event = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (
                    self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                move_event_to_upper_bound = current_location[max_zscore_bound_event]
            else:
                move_event_to_upper_bound = N
            if move_event_to_lower_bound == move_event_to_upper_bound:
                possible_positions = np.array([0])
            else:
                possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
            possible_sequences = np.zeros((len(possible_positions), N))
            possible_likelihood = np.full((len(possible_positions), 1), -np.inf)
            for index in range(len(possible_positions)):
                current_sequence = max_S
                # choose a position in the sequence to move an event to
                move_event_to = possible_positions[index]
                # move this event in its new position
                current_sequence = np.delete(current_sequence, move_event_from, 0)
                new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                # fit TEBM
                self.S = new_sequence
                self.means_ = self._get_means()
                self.covars_ = self.covars_prior
                self.fit()
                possible_likelihood[index] = self.score(self.X, self.lengths, self.jumps)
                possible_sequences[index, :] = self.S
            max_likelihood = max(possible_likelihood)
            max_S = possible_sequences[np.where(possible_likelihood == max_likelihood)[0][0]]
            if count<(N-1):
                print (str(round((count+1)/len(order_bio)*100,2))+'% complete')
        return max_S, max_likelihood

    def _seq_em(self, S, n_iter, seed_num):
        # parse out sequences by seed number
        S = np.array(S[seed_num])
        print ('Startpoint',seed_num)
        cur_seq = S
        cur_like = -np.inf
        flag = False
        for opt_i in range(int(n_iter)):
            print ('EM iteration',opt_i+1)
            seq, like = self._optimise_seq(cur_seq)
            print ('current', like, seq, 'max', cur_like, cur_seq)
            if like-cur_like < 1E-3:
                print ('EM converged in',opt_i+1,'iterations')
                flag = True
            elif like > cur_like:
                cur_seq = seq
                cur_like = like
            if flag:
                break
        return cur_seq, cur_like

    def _fit_tebm(self, n_zscores, z_max, n_start, n_iter, n_cores, cut_controls=False):
        # intialise z-score stuff
        z_val_arr = np.array([[x+1 for x in range(n_zscores)]]*self.n_features)
        z_max_arr = np.array([z_max]*self.n_features)
        IX_vals = np.array([[x for x in range(self.n_features)]*n_zscores]).T
        stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
        stage_zscore = np.array([y for x in z_val_arr.T for y in x])
        self.stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        self.stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
        self.min_biomarker_zscore = [0]*self.n_features
        self.max_biomarker_zscore = z_max_arr
        self.covars_prior = np.tile(np.identity(1), (self.n_components, self.n_features))
        # might want to fit sequence without controls
        if cut_controls:
            print ('Cutting controls from sequence fit!')
            X, lengths, jumps = [], [], []
            for i in range(len(self.lengths)):
                if labels[i] != 0:
                    nobs_i = self.lengths[i]
                    for x in self.X[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        X.append(x)
                    for x in self.jumps[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        jumps.append(x)
                    lengths.append(self.lengths[i])
            self.X = np.array(X)
            self.lengths = np.array(lengths)
            self.jumps = np.array(jumps)
        # do EM
        ml_seq_mat = np.zeros((1,self.stage_zscore.shape[1],n_start))
        ml_like_mat = np.zeros(n_start)
        if n_cores>1:
            pool = pathos.multiprocessing.ProcessingPool()
            pool.ncpus = n_cores
        else:
            # FIXME: serial version doesn't work
            #            pool = pathos.serial.SerialPool()
            pool = pathos.multiprocessing.ProcessingPool()
            pool.ncpus = n_cores
        # instantiate function as class to pass to pool.map
        # first calculate array of sequences - do this first or same random number will be used simultaneously on each processor
        # will return shape (n_start, 1)
        copier = partial(self._init_seq)
        # will return shape (n_start, 1)
        seq_mat = np.array(pool.map(copier, range(n_start)))
        # now optimise
        copier = partial(self._seq_em,
                         seq_mat[:,0],
                         n_iter)
        # will return shape (n_start, 2)
        par_mat = list(pool.map(copier, range(n_start)))
        # distribute to local matrices
        for i in range(n_start):
            ml_seq_mat[:, :, i] = par_mat[i][0]
            ml_like_mat[i] = par_mat[i][1]
        ix = np.argmax(ml_like_mat)
        ml_seq = ml_seq_mat[:, :, ix]
        ml_like = ml_like_mat[ix]
        # refit model on ML sequence
        self.S = ml_seq[0]
        self.covars_ = self.covars_prior
        self.means_ = self._get_means()
        self.fit()
        return ml_seq

class MultinomialTEBM(_BaseTEBM):
    r"""Hidden Markov Model with multinomial (discrete) emissions

    Parameters
    ----------

    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'e' for emissionprob.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Number of possible symbols emitted by the model (in the samples).

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    emissionprob\_ : array, shape (n_components, n_features)
        Probability of emitting a given symbol when in each state.

    Examples
    --------
    >>> from tebm.tebm import MultinomialTEBM
    >>> MultinomialTEBM(n_components=2)  #doctest: +ELLIPSIS
    MultinomialTEBM(algorithm='viterbi',...
    """
    # TODO: accept the prior on emissionprob_ for consistency.
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        _BaseTEBM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "e": nc * (nf - 1),
        }

    def _init(self, X, lengths=None):
        self._check_and_set_n_features(X)
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        if 'e' in self.init_params:
            self.emissionprob_ = self.random_state \
                .rand(self.n_components, self.n_features)
            normalize(self.emissionprob_, axis=1)

    def _check(self):
        super()._check()

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
        if self.emissionprob_.shape != (self.n_components, n_features):
            raise ValueError(
                "emissionprob_ must have shape (n_components, n_features)")
        else:
            self.n_features = n_features

    def _compute_log_likelihood(self, X):
        return log_mask_zero(self.emissionprob_)[:, np.concatenate(X)].T

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            for t, symbol in enumerate(np.concatenate(X)):
                stats['obs'][:, symbol] += posteriors[t]

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        if 'e' in self.params:
            self.emissionprob_ = (stats['obs']
                                  / stats['obs'].sum(axis=1)[:, np.newaxis])

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a Multinomial distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if hasattr(self, "n_features"):
            if self.n_features - 1 < X.max():
                raise ValueError(
                    "Largest symbol is {} but the model only emits "
                    "symbols up to {}"
                    .format(X.max(), self.n_features - 1))
        self.n_features = X.max() + 1


class GMMTEBM(_BaseTEBM):
    r"""Hidden Markov Model with Gaussian mixture emissions.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    n_mix : int
        Number of states in the GMM.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all mixture components of each state use **the same** full
          covariance matrix (note that this is not the same as for
          `MixtureTEBM`).

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    weights_prior : array, shape (n_mix, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`weights_`.

    means_prior, means_weight : array, shape (n_mix, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_mix, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    init_params : string, optional
        Controls which parameters are initialized prior to training. Can
        contain any combination of 's' for startprob, 't' for transmat, 'm'
        for means, 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    params : string, optional
        Controls which parameters are updated in the training process.  Can
        contain any combination of 's' for startprob, 't' for transmat, 'm' for
        means, and 'c' for covars, and 'w' for GMM mixing weights.
        Defaults to all parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    weights\_ : array, shape (n_components, n_mix)
        Mixture weights for each state.

    means\_ : array, shape (n_components, n_mix)
        Mean parameters for each mixture component in each state.

    covars\_ : array
        Covariance parameters for each mixture components in each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, n_mix)                          if "spherical",
            (n_components, n_mix, n_features)              if "diag",
            (n_components, n_mix, n_features, n_features)  if "full"
            (n_components, n_features, n_features)         if "tied",
    """

    def __init__(self, n_components=1, n_mix=1,
                 min_covar=1e-3, startprob_prior=1.0, transmat_prior=1.0,
                 weights_prior=1.0, means_prior=0.0, means_weight=0.0,
                 covars_prior=None, covars_weight=None,
                 algorithm="viterbi", covariance_type="diag",
                 random_state=None, n_iter=10, tol=1e-2,
                 verbose=False, params="stmcw",
                 init_params="stmcw"):
        _BaseTEBM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm, random_state=random_state,
                          n_iter=n_iter, tol=tol, verbose=verbose,
                          params=params, init_params=init_params)
        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.n_mix = n_mix
        self.weights_prior = weights_prior
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nm * nf,
            "c": {
                "spherical": nc * nm,
                "diag": nc * nm * nf,
                "full": nc * nm * nf * (nf + 1) // 2,
                "tied": nc * nf * (nf + 1) // 2,
            }[self.covariance_type],
            "w": nm - 1,
        }

    def _init(self, X, lengths=None):
        _check_and_set_gaussian_n_features(self, X)
        super()._init(X, lengths=lengths)
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # Default values for covariance prior parameters
        self._init_covar_priors()
        self._fix_priors_shape()

        main_kmeans = cluster.KMeans(n_clusters=nc,
                                     random_state=self.random_state)
        labels = main_kmeans.fit_predict(X)
        kmeanses = []
        for label in range(nc):
            kmeans = cluster.KMeans(n_clusters=nm,
                                    random_state=self.random_state)
            kmeans.fit(X[np.where(labels == label)])
            kmeanses.append(kmeans)

        if 'w' in self.init_params or not hasattr(self, "weights_"):
            self.weights_ = np.ones((nc, nm)) / (np.ones((nc, 1)) * nm)

        if 'm' in self.init_params or not hasattr(self, "means_"):
            self.means_ = np.stack(
                [kmeans.cluster_centers_ for kmeans in kmeanses])

        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(nf)
            if not cv.shape:
                cv.shape = (1, 1)
            if self.covariance_type == 'tied':
                self.covars_ = np.zeros((nc, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == 'full':
                self.covars_ = np.zeros((nc, nm, nf, nf))
                self.covars_[:] = cv
            elif self.covariance_type == 'diag':
                self.covars_ = np.zeros((nc, nm, nf))
                self.covars_[:] = np.diag(cv)
            elif self.covariance_type == 'spherical':
                self.covars_ = np.zeros((nc, nm))
                self.covars_[:] = cv.mean()

    def _init_covar_priors(self):
        if self.covariance_type == "full":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(1.0 + self.n_features + 1.0)
        elif self.covariance_type == "tied":
            if self.covars_prior is None:
                self.covars_prior = 0.0
            if self.covars_weight is None:
                self.covars_weight = -(self.n_mix + self.n_features + 1.0)
        elif self.covariance_type == "diag":
            if self.covars_prior is None:
                self.covars_prior = -1.5
            if self.covars_weight is None:
                self.covars_weight = 0.0
        elif self.covariance_type == "spherical":
            if self.covars_prior is None:
                self.covars_prior = -(self.n_mix + 2.0) / 2.0
            if self.covars_weight is None:
                self.covars_weight = 0.0

    def _fix_priors_shape(self):
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        # If priors are numbers, this function will make them into a
        # matrix of proper shape
        self.weights_prior = np.broadcast_to(
            self.weights_prior, (nc, nm)).copy()
        self.means_prior = np.broadcast_to(
            self.means_prior, (nc, nm, nf)).copy()
        self.means_weight = np.broadcast_to(
            self.means_weight, (nc, nm)).copy()

        if self.covariance_type == "full":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm, nf, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm)).copy()
        elif self.covariance_type == "tied":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nf, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, nc).copy()
        elif self.covariance_type == "diag":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm, nf)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm, nf)).copy()
        elif self.covariance_type == "spherical":
            self.covars_prior = np.broadcast_to(
                self.covars_prior, (nc, nm)).copy()
            self.covars_weight = np.broadcast_to(
                self.covars_weight, (nc, nm)).copy()

    def _check(self):
        super()._check()
        if not hasattr(self, "n_features"):
            self.n_features = self.means_.shape[2]
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        self._init_covar_priors()
        self._fix_priors_shape()

        # Checking covariance type
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError("covariance_type must be one of {}"
                             .format(COVARIANCE_TYPES))

        self.weights_ = np.array(self.weights_)
        # Checking mixture weights' shape
        if self.weights_.shape != (nc, nm):
            raise ValueError("mixture weights must have shape "
                             "(n_components, n_mix), actual shape: {}"
                             .format(self.weights_.shape))

        # Checking mixture weights' mathematical correctness
        if not np.allclose(np.sum(self.weights_, axis=1), np.ones(nc)):
            raise ValueError("mixture weights must sum up to 1")

        # Checking means' shape
        self.means_ = np.array(self.means_)
        if self.means_.shape != (nc, nm, nf):
            raise ValueError("mixture means must have shape "
                             "(n_components, n_mix, n_features), "
                             "actual shape: {}".format(self.means_.shape))

        # Checking covariances' shape
        self.covars_ = np.array(self.covars_)
        covars_shape = self.covars_.shape
        needed_shapes = {
            "spherical": (nc, nm),
            "tied": (nc, nf, nf),
            "diag": (nc, nm, nf),
            "full": (nc, nm, nf, nf),
        }
        needed_shape = needed_shapes[self.covariance_type]
        if covars_shape != needed_shape:
            raise ValueError("{!r} mixture covars must have shape {}, "
                             "actual shape: {}"
                             .format(self.covariance_type,
                                     needed_shape, covars_shape))

        # Checking covariances' mathematical correctness
        from scipy import linalg

        if (self.covariance_type == "spherical" or
                self.covariance_type == "diag"):
            if np.any(self.covars_ < 0):
                raise ValueError("{!r} mixture covars must be non-negative"
                                 .format(self.covariance_type))
            if np.any(self.covars_ == 0):
                _log.warning("Degenerate mixture covariance")
        elif self.covariance_type == "tied":
            for i, covar in enumerate(self.covars_):
                if not np.allclose(covar, covar.T):
                    raise ValueError("Covariance of state #{} is not symmetric"
                                     .format(i))
                min_eigvalsh = np.linalg.eigvalsh(covar).min()
                if min_eigvalsh < 0:
                    raise ValueError("Covariance of state #{} is not positive "
                                     "definite".format(i))
                if min_eigvalsh == 0:
                    _log.warning("Covariance of state #%d has a null "
                                 "eigenvalue.", i)
        elif self.covariance_type == "full":
            for i, mix_covars in enumerate(self.covars_):
                for j, covar in enumerate(mix_covars):
                    if not np.allclose(covar, covar.T):
                        raise ValueError(
                            "Covariance of state #{}, mixture #{} is not "
                            "symmetric".format(i, j))
                    min_eigvalsh = np.linalg.eigvalsh(covar).min()
                    if min_eigvalsh < 0:
                        raise ValueError(
                            "Covariance of state #{}, mixture #{} is not "
                            "positive definite".format(i, j))
                    if min_eigvalsh == 0:
                        _log.warning("Covariance of state #%d, mixture #%d "
                                     "has a null eigenvalue.", i, j)

    def _generate_sample_from_state(self, state, random_state=None):
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        cur_weights = self.weights_[state]
        i_gauss = random_state.choice(self.n_mix, p=cur_weights)
        if self.covariance_type == 'tied':
            # self.covars_.shape == (n_components, n_features, n_features)
            # shouldn't that be (n_mix, ...)?
            covs = self.covars_
        else:
            covs = self.covars_[:, i_gauss]
            covs = fill_covars(covs, self.covariance_type,
                               self.n_components, self.n_features)
        return random_state.multivariate_normal(
            self.means_[state, i_gauss], covs[state]
        )

    def _compute_log_weighted_gaussian_densities(self, X, i_comp):
        cur_means = self.means_[i_comp]
        cur_covs = self.covars_[i_comp]
        if self.covariance_type == 'spherical':
            cur_covs = cur_covs[:, np.newaxis]
        log_cur_weights = np.log(self.weights_[i_comp])

        return log_multivariate_normal_density(
            X, cur_means, cur_covs, self.covariance_type
        ) + log_cur_weights

    def _compute_log_likelihood(self, X):
        n_samples, _ = X.shape
        res = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, i)
            with np.errstate(under="ignore"):
                res[:, i] = logsumexp(log_denses, axis=1)

        return res

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['n_samples'] = 0
        stats['post_comp_mix'] = None
        stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
        stats['post_sum'] = np.zeros(self.n_components)
        stats['samples'] = None
        stats['centered'] = None
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          post_comp, fwdlattice, bwdlattice):

        # TODO: support multiple frames

        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
        )

        n_samples, _ = X.shape

        stats['n_samples'] = n_samples
        stats['samples'] = X

        post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
        for p in range(self.n_components):
            log_denses = self._compute_log_weighted_gaussian_densities(X, p)
            log_normalize(log_denses, axis=-1)
            with np.errstate(under="ignore"):
                post_mix[:, p, :] = np.exp(log_denses)

        with np.errstate(under="ignore"):
            post_comp_mix = post_comp[:, :, np.newaxis] * post_mix
        stats['post_comp_mix'] = post_comp_mix

        stats['post_mix_sum'] = np.sum(post_comp_mix, axis=0)
        stats['post_sum'] = np.sum(post_comp, axis=0)

        stats['centered'] = X[:, np.newaxis, np.newaxis, :] - self.means_

    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        nc = self.n_components
        nf = self.n_features
        nm = self.n_mix

        n_samples = stats['n_samples']

        # Maximizing weights
        alphas_minus_one = self.weights_prior - 1
        new_weights_numer = stats['post_mix_sum'] + alphas_minus_one
        new_weights_denom = (
            stats['post_sum'] + np.sum(alphas_minus_one, axis=1)
        )[:, np.newaxis]
        new_weights = new_weights_numer / new_weights_denom

        # Maximizing means
        lambdas, mus = self.means_weight, self.means_prior
        new_means_numer = (
            np.einsum('ijk,il->jkl', stats['post_comp_mix'], stats['samples'])
            + lambdas[:, :, np.newaxis] * mus
        )
        new_means_denom = (stats['post_mix_sum'] + lambdas)[:, :, np.newaxis]
        new_means = new_means_numer / new_means_denom

        # Maximizing covariances
        centered_means = self.means_ - mus

        if self.covariance_type == 'full':
            centered = stats['centered'].reshape((n_samples, nc, nm, nf, 1))
            centered_t = stats['centered'].reshape((n_samples, nc, nm, 1, nf))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((nc, nm, nf, 1))
            centr_means_resh_t = centered_means.reshape((nc, nm, 1, nf))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            new_cov_numer = (
                np.einsum(
                    'ijk,ijklm->jklm', stats['post_comp_mix'], centered_dots)
                + psis_t
                + lambdas[:, :, np.newaxis, np.newaxis] * centered_means_dots
            )
            new_cov_denom = (
                stats['post_mix_sum'] + 1 + nus + nf + 1
            )[:, :, np.newaxis, np.newaxis]
            new_cov = new_cov_numer / new_cov_denom

        elif self.covariance_type == 'diag':
            centered2 = stats['centered'] ** 2
            centered_means2 = centered_means ** 2

            alphas = self.covars_prior
            betas = self.covars_weight

            new_cov_numer = (
                np.einsum('ijk,ijkl->jkl', stats['post_comp_mix'], centered2)
                + lambdas[:, :, np.newaxis] * centered_means2
                + 2 * betas
            )
            new_cov_denom = (
                stats['post_mix_sum'][:, :, np.newaxis] + 1 + 2 * (alphas + 1)
            )
            new_cov = new_cov_numer / new_cov_denom

        elif self.covariance_type == 'spherical':
            centered_norm2 = np.sum(stats['centered'] ** 2, axis=-1)

            alphas = self.covars_prior
            betas = self.covars_weight

            centered_means_norm2 = np.sum(centered_means ** 2, axis=-1)

            new_cov_numer = (
                np.einsum(
                    'ijk,ijk->jk', stats['post_comp_mix'], centered_norm2)
                + lambdas * centered_means_norm2
                + 2 * betas
            )
            new_cov_denom = nf * (stats['post_mix_sum'] + 1) + 2 * (alphas + 1)
            new_cov = new_cov_numer / new_cov_denom

        elif self.covariance_type == 'tied':
            centered = stats['centered'].reshape((n_samples, nc, nm, nf, 1))
            centered_t = stats['centered'].reshape((n_samples, nc, nm, 1, nf))
            centered_dots = centered * centered_t

            psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
            nus = self.covars_weight

            centr_means_resh = centered_means.reshape((nc, nm, nf, 1))
            centr_means_resh_t = centered_means.reshape((nc, nm, 1, nf))
            centered_means_dots = centr_means_resh * centr_means_resh_t

            lambdas_cmdots_prod_sum = (
                np.einsum('ij,ijkl->ikl', lambdas, centered_means_dots))

            new_cov_numer = (
                np.einsum(
                    'ijk,ijklm->jlm', stats['post_comp_mix'], centered_dots)
                + lambdas_cmdots_prod_sum + psis_t)
            new_cov_denom = (
                stats['post_sum'] + nm + nus + nf + 1
            )[:, np.newaxis, np.newaxis]
            new_cov = new_cov_numer / new_cov_denom

        # Assigning new values to class members
        self.weights_ = new_weights
        self.means_ = new_means
        self.covars_ = new_cov
