# Variable time interval Temporal Event-Based Model
# Base class
# Author: Peter Wijeratne (p.wijeratne@sussex.ac.uk)

import logging
import string
import sys
from collections import deque

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn import cluster
from sklearn.utils import check_array, check_random_state

import scipy as sp

from . import _tebmc_var, _utils
from .utils import normalize, log_normalize, iter_from_X_lengths, log_mask_zero

from kde_ebm.mixture_model import get_prob_mat

_log = logging.getLogger(__name__)
#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))

import warnings

class ConvergenceMonitor:
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.

    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.

    >>> from tebm.base import ConvergenceMonitor
    >>> from tebm import tebm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = tebm.GaussianTEBM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        # XXX we might want to check that ``logprob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.tol))


class _BaseTEBM(BaseEstimator):
    r"""Base class for Hidden Markov Models.

    This class allows for easy evaluation of, sampling from, and
    maximum a posteriori estimation of the parameters of a TEBM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

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
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    """
    def __init__(self, X=None, lengths=None, jumps=None,
                 n_components=1, time_mean=1.0, startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None, n_iter=100,
                 tol=1e-3, verbose=False, params=string.ascii_letters,
                 init_params=string.ascii_letters, allow_nan=False,
                 fwd_only=False, order=None):
        self.X = X
        self.lengths = lengths
        self.jumps = jumps
        self.n_components = n_components
        self.time_mean = time_mean
        self.n_obs = X.shape[0]
        self.n_features = X.shape[1]
        self.params = params
        self.init_params = init_params
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        self.allow_nan = allow_nan
        # step sizes
        self.step_sizes = np.unique(jumps).astype(float)
        self.step_sizes = self.step_sizes[self.step_sizes>0]
        self.fwd_only = fwd_only
        self.order = order
        # normalise to smallest stepsize, so the base unit = 1
        #FIXME: this will change the time units if the base unit != 1
        #        if self.step_sizes[0]!=1:
        #            print ('Renormalising timescale!')
        #            self.step_sizes /= self.step_sizes[0]
        # initialise startprob
        if startprob_prior is None:
            self.startprob_prior = np.full(self.n_components, 1./self.n_components)
        else:
            self.startprob_prior = startprob_prior
        self.startprob_ = self.startprob_prior
        # initialise Q_prior and Q_
        self.Q_prior = np.empty((self.n_components, self.n_components)).astype(complex)
        for i in range(self.n_components):
            #            vec = np.random.random(self.n_components-1)
            vec = np.ones(self.n_components-1)
            #            epsilon = np.random.rand(self.n_components-1)
            #            vec = np.ones(self.n_components-1) + epsilon#np.random.rand(self.n_components-1)
            vec /= np.sum(vec)
            self.Q_prior[i,:i] = vec[:i]
            self.Q_prior[i,i+1:] = vec[i:]
            #FIXME: set time scale externally
            self.Q_prior[i,i] = -self.time_mean#-.25#np.log(1/self.n_components)
            #            self.Q_prior[i,i] = -np.sum(self.Q_prior[i])
        # zero-out forbidden states
        if self.fwd_only:
            for i in range(self.n_components):
                for j in range(self.n_components):
                    if j<i:
                        self.Q_prior[i,j] = 0            
        if self.order:
            for i in range(self.n_components):
                for j in range(self.n_components):
                    if not (j<=(i+self.order) and j>=(i-self.order)):
                        self.Q_prior[i,j] = 0
        # renormalise
        for i in range(self.n_components):
            scale = np.sum([x if jj!=i else 0 for jj,x in enumerate(self.Q_prior[i])])
#            if i==(self.n_components-1) and self.fwd_only:
#                self.Q_prior[i,i] = -1E-9
#                self.Q_prior[i,:i] = 1E-9/(self.n_components-1)
#                break
            for j in range(self.n_components):
                if i!=j:
                    if scale!=0:
                        self.Q_prior[i,j] *= -self.Q_prior[i,i]/scale
                    else:
                        self.Q_prior[i,j] = 0.
                elif i==(self.n_components-1) and j==(self.n_components-1) and self.fwd_only:
                    self.Q_prior[i,j] = 0.
        #
#        for i in range(self.n_components):
#            for j in range(self.n_components):
#                if i==j:
#                    self.Q_prior[i,j] -= 1E-9
#                else:
#                    self.Q_prior[i,j] += 1E-9/(self.n_components-1)
#        print ('Q_prior', self.Q_prior)
        # check Q_prior        
        if np.sum(self.Q_prior)>1E-9:
            print ('Q_prior not initialised correctly!',np.sum(self.Q_prior),self.Q_prior.astype(float))
            quit()
        #        print ('Q_prior', self.Q_prior)
        self.Q_ = self.Q_prior
        # initialise transmat_prior and transmat_
        if len(self.step_sizes):
            self.transmat_prior = []
            for i in range(len(self.step_sizes)):
                self.transmat_prior.append(sp.linalg.expm(self.Q_prior*self.step_sizes[i]).astype(float))
                #NEW START
#                self.transmat_prior[i][:-1,:][np.where(self.Q_prior[:-1,:]==0.)] = 0.
#                normalize(self.transmat_prior[i], axis=1)
                #NEW END
            self.transmat_prior = np.array(self.transmat_prior)
            self.transmat_ = np.array([self.transmat_prior[i] for i in range(len(self.step_sizes))])
        else:
            self.transmat_prior = np.full((self.n_components, self.n_components), 1./self.n_components)
            self.transmat_ = self.transmat_prior
        # if Q_prior is not diagonalisable, then have to use full matrix exponential method
        if np.any([self.Q_prior[i,i]==0 for i in range(self.n_components)]):
            self.eigen = False
            #            if verbose:
            print ('Transition rate matrix is not diagonalisable! Using matrix exponential...(slower)')
        else:
            self.eigen = True
            #            if verbose:
            print ('Transition rate matrix is diagonalisable! Using eigendecomposition...(faster)')
        #        print ('transmat_prior', self.transmat_prior)        
    def get_stationary_distribution(self):
        """Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        _utils.check_is_fitted(self, "transmat_")
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def score_samples(self, X, lengths=None, jumps=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()
        if self.allow_nan:
            X = check_array(X, force_all_finite='allow-nan')
        else:
            X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        posteriors = np.zeros((n_samples, self.n_components))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j], i, j)
            logprobij, fwdlattice = self._do_forward_pass(framelogprob, jumps[i:j])
            logprob += logprobij
            bwdlattice = self._do_backward_pass(framelogprob, jumps[i:j])
            posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    def score(self, X, lengths=None, jumps=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()
        if self.allow_nan:
            X = check_array(X, force_all_finite='allow-nan')
        else:
            X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j], i, j)
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob, jumps[i:j])
            logprob += logprobij
        return logprob

    def _decode_viterbi(self, X, i, j):
        framelogprob = self._compute_log_likelihood(X, i, j)
        return self._do_viterbi_pass(framelogprob, self.jumps[i:j])
    
    def _decode_map(self, X, i, j):
        # FIXME: urgh this is bad coding
        _, posteriors = self.score_samples(X, np.array([1]), self.jumps[i:j])
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, X, lengths=None, jumps=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]
        if self.allow_nan:
            X = check_array(X, force_all_finite='allow-nan')
        else:
            X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j], i, j)
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, lengths=None, jumps=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """

    def predict_proba(self, X, lengths=None, jumps=None):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        
    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.

        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)
        #FIXME: should sample from Q instead of picking arbitrary A
        #        print ('FIXME! base_var:sample')
        transmat_cdf = np.cumsum(self.transmat_[0], axis=1)

        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.allow_nan:
            X = check_array(self.X, force_all_finite='allow-nan')
        else:
            X = check_array(self.X)
        # FIXME: these arguments are only necessary for CTHMM
        self._init(self.X, self.lengths, self.jumps)
        self._check()
        self.monitor_._reset()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, self.lengths):
                framelogprob = self._compute_log_likelihood(X[i:j], i, j)
                framejumps = self.jumps[i:j]
                logprob, fwdlattice = self._do_forward_pass(framelogprob, framejumps)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob, framejumps)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, self.X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice, framejumps)
            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)
            #
            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break
            #
            if (self.transmat_.sum(axis=1) == 0).any():
                _log.warning("Some rows of transmat_ have zero sum because no "
                             "transition from the state was ever observed.")
        return self
    
    def _do_viterbi_pass(self, framelogprob, framejumps):
        n_samples, n_components = framelogprob.shape
        # match the correct transition matrix for sample t
        log_transmat_tau = []
        # FIXME: guard against single samples - this is a dummy
        if n_samples==1:
            log_transmat_tau.append(self.transmat_[0])
        else:
            for t in range(1,len(framejumps)):
                # NOTE: this is different to _do_forward_pass, _do_backward_pass, _accumulate_sufficient_statistics
                # in those functions we lookup the transmat_ for a given framejump
                # here we might be staging data we've not seen before, which means the transmat_ might not have been stored
                # in this case we have to use Q to calculate the transmat_ for this interval
                log_transmat_tau.append(log_mask_zero(np.real(sp.linalg.expm(framejumps[t]*self.Q_))))
        log_transmat_tau = np.array(log_transmat_tau)
        #
        state_sequence, logprob = _tebmc_var._viterbi(n_samples,
                                                      n_components,
                                                      log_mask_zero(self.startprob_),            
                                                      log_transmat_tau,
                                                      framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob, framejumps):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        # match the correct transition matrix for sample t
        log_transmat_tau = []
        # FIXME: guard against single samples - this is a dummy
        if n_samples==1:
            log_transmat_tau.append(self.transmat_[0])
        else:
            for t in range(1,len(framejumps)):
                log_transmat_tau.append(log_mask_zero(self.transmat_[np.where(framejumps[t]==self.step_sizes)[0][0]]))
        log_transmat_tau = np.array(log_transmat_tau)
        _tebmc_var._forward(n_samples,
                            n_components,
                            log_mask_zero(self.startprob_),
                            log_transmat_tau,
                            framelogprob,
                            fwdlattice)
        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice
    
    def _do_backward_pass(self, framelogprob, framejumps):
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        # match the correct transition matrix for sample t
        log_transmat_tau = []
        # FIXME: guard against single samples - this is a dummy
        if n_samples==1:
            log_transmat_tau.append(self.transmat_[0])
        else:
            for t in range(1,len(framejumps)):
                log_transmat_tau.append(log_mask_zero(self.transmat_[np.where(framejumps[t]==self.step_sizes)[0][0]]))
        log_transmat_tau = np.array(log_transmat_tau)
        _tebmc_var._backward(n_samples,
                             n_components,
                             log_mask_zero(self.startprob_),
                             log_transmat_tau,
                             framelogprob,
                             bwdlattice)
        return bwdlattice

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _init(self, X, lengths):
        """
        # set in child class
        """
    
    def _check(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))
        
        self.transmat_ = np.asarray(self.transmat_)
        for i in range(self.transmat_.shape[0]):
            if self.transmat_[i].shape != (self.n_components, self.n_components):
                raise ValueError(
                    "transmat_ must have shape (n_components, n_components)")
            if not np.allclose(self.transmat_[i].sum(axis=1), 1.0):
                warnings.warn("rows of transmat_ must sum to 1.0")
                print (self.transmat_[i])

    def _compute_log_likelihood(self, i, j):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """

    def _generate_sample_from_state(self, state, random_state=None):
        """Generates a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.

        random_state: RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.

        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.

        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'C_soft': np.zeros((len(self.step_sizes), self.n_components, self.n_components)),
                 'C_hard': np.zeros((len(self.step_sizes), self.n_components, self.n_components))}
        return stats
    
    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors,
                                          fwdlattice, bwdlattice, framejumps):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseTEBM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.
        
        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return
            # update soft count matrix for given time interval
            # FIXME: double check transmat matches fwdlattice, bwdlattice
            for t in range(1,len(framejumps)):
                log_xi_sum = np.full((n_components, n_components), -np.inf)
                jump_idx = np.where(framejumps[t]==self.step_sizes)[0][0]
                _tebmc_var._compute_log_xi_sum(2,
                                               n_components,
                                               fwdlattice[t-1:t+1],
                                               log_mask_zero(self.transmat_[jump_idx]),
                                               bwdlattice[t-1:t+1],
                                               framelogprob[t-1:t+1], #0:2, 1:3
                                               log_xi_sum)
                with np.errstate(under="ignore"):
                    # soft count - posterior probability
                    C_soft = np.exp(log_xi_sum)
                    # posterior should sum to 1 over all transitions
                    normalize(C_soft)
                    stats['C_soft'][jump_idx] += C_soft
                    # hard count - MAP stage
                    C_hard = np.zeros((n_components, n_components))
                    max_loc = np.unravel_index(C_soft.argmax(), C_soft.shape)
                    C_hard[max_loc[0],max_loc[1]] = 1.
                    stats['C_hard'][jump_idx] += C_hard 

    def drawState(self, posteriors, sample=False):
        # draw a state from the posterior, via either sampling or MAP
        if sample:
            cdf = np.cumsum(posteriors, axis=1)
            r = np.random.uniform(size=len(posteriors)) * cdf[:,-1]
            drawn_state = np.zeros(len(posteriors))
            for n in range(len(posteriors)):
                drawn_state[n] = np.searchsorted(cdf[n,:], r[n])
        else:
            drawn_state = np.argmax(posteriors, axis=1)
        return drawn_state

    def compute_Q_eigen(self, stats, soft=True):
        # eigen method for calculating Q - see https://md2k.org/images/papers/methods/LearningC-THMM.pdf
        step_sizes = self.step_sizes
        n_states = self.n_components
        if soft:
            # soft count frequency matrix
            C = stats['C_soft']
        else:
            # hard count frequency matrix
            C = stats['C_hard']
        # scale C by transition matrix for given interval
        for t in range(len(step_sizes)):
            for i in range(n_states):
                for j in range(n_states):
                    if self.transmat_[t,i,j] > 0:
                        C[t,i,j] /= self.transmat_[t,i,j]
                    else:
                        C[t,i,j] = 0.
        E_t = np.zeros(n_states, dtype=complex)
        E_n = np.zeros((n_states,n_states), dtype=complex)
        Q = np.zeros((n_states,n_states), dtype=complex)
        lambdas, U = np.linalg.eig(self.Q_.astype(complex))
        V = np.linalg.inv(U)
        for t in range(len(step_sizes)):
            tau = step_sizes[t]
            Psi = np.zeros((n_states,n_states), dtype=complex)
            for i in range(Psi.shape[0]):
                for j in range(Psi.shape[0]):
                    if lambdas[i]==lambdas[j]:
                        Psi[i,j] = tau*np.exp(tau*lambdas[i])
                    else:
                        Psi[i,j] = (np.exp(tau*lambdas[i])-np.exp(tau*lambdas[j]))/(lambdas[i]-lambdas[j])
            B = np.matmul(np.matmul(U.T,C[t]),V.T)
            for i in range(n_states):
                A = np.matmul(V[:,i].reshape(n_states,1),U[i].reshape(1,n_states))*Psi
                E_t_i = np.sum(A*B)
                if E_t_i < 0:
                    continue
                E_t[i] += E_t_i
                for j in range(n_states):
                    if i==j:
                        continue
                    A = np.matmul(V[:,i].reshape(n_states,1),U[j].reshape(1,n_states))*Psi
                    E_n_ij = self.Q_[i,j]*np.sum(A*B)
                    if (E_n_ij) < 0:
                        continue
                    E_n[i,j] += E_n_ij
        Q = E_n/E_t[:,None]
        for i in range(Q.shape[0]):
            Q[i,i] = -np.sum([x if j!=i else 0 for j,x in enumerate(Q[i])])
        return Q

    def expm_A(self, tau, i, j):
        # calculate matrix of matrices - see https://md2k.org/images/papers/methods/LearningC-THMM.pdf
        n_states = self.n_components
        I = np.zeros((n_states,n_states))
        I[i,j] = 1.
        A = np.zeros((2*n_states,2*n_states)).astype(complex)
        A[:n_states,:n_states] = self.Q_
        A[:n_states:,n_states:] = I
        A[n_states:,:n_states] = np.zeros((n_states,n_states))
        A[n_states:,n_states:] = self.Q_
        expm_A = sp.linalg.expm(tau*A).astype(float)[:n_states,n_states:]
        return expm_A
    
    def compute_Q_expm(self, stats, soft=True):
        # expm method for calculating Q - see https://md2k.org/images/papers/methods/LearningC-THMM.pdf
        step_sizes = self.step_sizes
        n_states = self.n_components
        if soft:
            # soft count frequency matrix
            C = stats['C_soft']
        else:
            # hard count frequency matrix
            C = stats['C_hard']
#        print ('C',C.astype(int))
        #########################################################################################################################TESTING
        #        for i in range(C.shape[0]):
        #            C[i] = np.maximum(self.transmat_prior[i] - 1 + C[i], 0)        
        #########################################################################################################################TESTING
        E_R = np.zeros(n_states)
        E_N = np.zeros((n_states,n_states))
        for t in range(len(step_sizes)):
            tau = step_sizes[t]
            for i in range(n_states):
                for j in range(n_states):
                    N_ij = self.expm_A(tau,i,j)
                    #FIXME: do this without the loops
                    for k in range(n_states):
                        for l in range(n_states):
                            if i==j:
                                E_R_i = C[t,k,l]*N_ij[k,l]
                                if self.transmat_[t,k,l] > 0:
#                                if self.transmat_[t,k,l] > np.finfo(np.double).tiny:
                                    E_R[i] += E_R_i/self.transmat_[t,k,l]
                            else:
                                E_N_ij = self.Q_[i,j]*C[t,k,l]*N_ij[k,l]
                                if self.transmat_[t,k,l] > 0:
#                                if self.transmat_[t,k,l] > np.finfo(np.double).tiny:
#                                    print (E_N_ij, self.transmat_[t,k,l])
                                    E_N[i,j] += E_N_ij/self.transmat_[t,k,l]
#        print (E_N)
#        print (E_R)
        #FIXME: HACK to prevent zero entries in E_R causing NaNs in Q; find cause
        # can reproduce using noisy data, which can produce elements of E_R = 0 because the EM algorithm pushes everyone out of a stage (over-fits)
        if np.any(E_R==0):
            E_R[E_R==0] = np.inf
            print ('WARNING! there is at least one stage with zero initial probability')
        #
        Q = E_N/E_R[:,None].astype(complex)
        for i in range(Q.shape[0]):
            Q[i,i] = -np.sum([x if j!=i else 0 for j,x in enumerate(Q[i])])
#        print ('Q',Q)
#        print ('P',sp.sparse.linalg.expm(Q.astype(float)))
        return Q
    
    def _do_mstep(self, stats):
        """Performs the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # If a prior is < 1, `prior - 1 + starts['start']` can be negative.  In
        # that case maximization of (n1+e1) log p1 + ... + (ns+es) log ps under
        # the conditions sum(p) = 1 and all(p >= 0) show that the negative
        # terms can just be set to zero.
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right TEBM.
        if 's' in self.params:
            startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                    0)
            self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
            normalize(self.startprob_)
#            from matplotlib import pyplot as plt
#            fig, ax = plt.subplots()
#            ax.bar(np.arange(self.startprob_.shape[0]),self.startprob_)
#            plt.show()
        if 't' in self.params:
            # calculate rate matrix
            if self.eigen:
                self.Q_ = self.compute_Q_eigen(stats)
            else:
                self.Q_ = self.compute_Q_expm(stats)
            # calculate transition matrix for each time interval and apply prior
            for i in range(len(self.step_sizes)):
                self.transmat_[i] = sp.linalg.expm(self.step_sizes[i]*self.Q_)
                # apply prior
                self.transmat_[i][np.where(self.transmat_prior[i]==0.)] = 0.
                #NEW START
                #                self.transmat_[i][:-1,:][np.where(self.Q_prior[:-1,:]==0.)] = 0.
                #NEW END
                # guard against small -ve numbers from matrix exponential
                #                self.transmat_[i][np.where(self.transmat_[i]<0.)] = 0.
                normalize(self.transmat_[i], axis=1)

                
