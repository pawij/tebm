# Fixed time interval Temporal Event-Based Model
# Derived class from base_fix.py
# Author: Peter Wijeratne (p.wijeratne@pm.me)

import numpy as np
from scipy.special import logsumexp
from functools import partial
import pathos

from .base_fix import BaseTEBM
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models, get_prob_mat

class MixtureTEBM(BaseTEBM):

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
        
        BaseTEBM.__init__(self,
                          X=X,
                          lengths=lengths, 
                          n_stages=n_stages,
                          time_mean=time_mean,
                          n_iter=n_iter,
                          fwd_only=fwd_only,
                          order=order,
                          algo=algo,
                          verbose=verbose)
    
    def compute_log_likelihood(self, X, start_i, end_i):
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

    def stages(self, X, lengths=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.prob_mat = get_prob_mat(X, self.mixtures)
        ###
        stage_sequence = self.stage_X(X, lengths)
        return stage_sequence

    def posteriors(self, X, lengths=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        self.prob_mat = get_prob_mat(X, self.mixtures)
        ###
        posteriors = self.posteriors_X(X, lengths)
        return posteriors
    
    def gen_sample(self, stage):
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
        ###
        return np.random.multivariate_normal(self.means[stage], self.covars[stage])

    def optimise_seq(self, S):
        N = self.n_features
        max_S = S.copy()
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
                # fit TEBM
                self.S = new_sequence
                self.fit()
                possible_likelihood[index] = self.compute_model_log_likelihood(self.X, self.lengths)
                possible_sequences[index, :] = self.S
            max_likelihood = max(possible_likelihood)
            max_S = possible_sequences[np.where(possible_likelihood == max_likelihood)[0][0]]
            if count<(N-1):
                print (str(round((count+1)/len(order_bio)*100,2))+'% complete')
        return max_S, max_likelihood

    def seq_em(self, S, n_iter, seed_num):
        # parse out sequences by seed number
        S = np.array(S[seed_num])
        print ('Startpoint',seed_num)
        cur_seq = S
        cur_like = -np.inf
        flag = False
        for opt_i in range(int(n_iter)):
            print ('EM iteration',opt_i+1)
            seq, like = self.optimise_seq(cur_seq)
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

    def fit_tebm(self, labels, n_start, n_iter, n_cores, model_type='GMM', constrained=False, cut_controls=False):
        # only use baseline data to fit mixture models
        X0 = []
        for i in range(len(self.lengths)):
            X0.append(self.X[np.sum(self.lengths[:i])])
        X0 = np.array(X0)
        if model_type == 'KDE':
            mixtures = fit_all_kde_models(X0, labels)
        else:
            mixtures = fit_all_gmm_models(X0, labels)#, constrained)
        # might want to fit sequence without controls
        if cut_controls:
            print ('Cutting controls from sequence fit!')
            X, lengths = [], []
            for i in range(len(self.lengths)):
                if labels[i] != 0:
                    nobs_i = self.lengths[i]
                    for x in self.X[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        X.append(x)
                    lengths.append(self.lengths[i])
            self.X = np.array(X)
            self.lengths = np.array(lengths)
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
        copier = partial(self.init_seq)
        # will return shape (n_start, 1)
        seq_mat = np.array(pool.map(copier, range(n_start)))
        # now optimise
        copier = partial(self.seq_em,
                         seq_mat[:,0],
                         n_iter)
        # will return shape (n_start, 2)
        par_mat = np.array(pool.map(copier, range(n_start)))
        # distribute to local matrices
        for i in range(n_start):
            ml_seq_mat[:, :, i] = par_mat[i, 0]
            ml_like_mat[i] = par_mat[i, 1]
        ix = np.argmax(ml_like_mat)
        ml_seq = ml_seq_mat[:, :, ix]
        ml_like = ml_like_mat[ix]
        # refit model on ML sequence
        self.S = ml_seq[0]
        self.fit()
        return ml_seq, self.mixtures

    def init_seq(self, seed_num):
        #FIXME: issue with seeding by seed_num is that every time you call fit_tebm, it will initialise the same sequences
        # ensure randomness across parallel processes
        np.random.seed(seed_num)
        S = np.arange(self.n_features)
        np.random.shuffle(S)
        return [S]

class ZscoreTEBM(BaseTEBM):

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
        
        BaseTEBM.__init__(self,
                          X=X,
                          lengths=lengths, 
                          n_stages=n_stages,
                          time_mean=time_mean,
                          n_iter=n_iter,
                          fwd_only=fwd_only,
                          order=order,
                          algo=algo,
                          verbose=verbose)
        
    def compute_log_likelihood(self, X, start_i, end_i):
        n_samples, n_dim = X.shape
        return -0.5 * (n_dim * np.log(2 * np.pi)
                       + np.log(self.covars).sum(axis=-1)
                       + ((X[:, None, :] - self.means) ** 2 / self.covars).sum(axis=-1))

    def stages(self, X, lengths=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        ###
        stage_sequence = self.stage_X(X, lengths)
        return stage_sequence

    def posteriors(self, X, lengths=None):
        ### FIXME: is there a general way of doing this?
        self.X = X
        self.lengths = lengths
        ###
        posteriors = self.posteriors_X(X, lengths)
        return posteriors

    def gen_sample(self, n_samples=1):
        p_vec_cdf = np.cumsum(self.p_vec)
        a_mat_cdf = np.cumsum(self.a_mat, axis=1)
        X_sample, k_sample = [], []
        k_i = (p_vec_cdf > np.random.rand()).argmax()
        for i in range(n_samples):
            k_i = (a_mat_cdf[k_i] > np.random.rand()).argmax()
            k_sample.append(k_i)
            #            X_sample.append(np.random.multivariate_normal(self.means[k_i], self.covars[k_i]))
            #FIXME: make multivariate
            X_sample.append(np.random.normal(self.means[k_i], self.covars[k_i]))
        return np.array(X_sample), np.array(k_sample)

    def init_seq(self, seed_num):
        np.random.seed(seed_num)    
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
    
    def get_means(self):
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

    def optimise_seq(self, S):
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
                self.means = self.get_means()
                self.covars = self.covars_prior
                self.fit()
                possible_likelihood[index] = self.compute_model_log_likelihood(self.X, self.lengths)
                possible_sequences[index, :] = self.S
            max_likelihood = max(possible_likelihood)
            max_S = possible_sequences[np.where(possible_likelihood == max_likelihood)[0][0]]
            if count<(N-1):
                print (str(round((count+1)/len(order_bio)*100,2))+'% complete')
        return max_S, max_likelihood

    def seq_em(self, S, n_iter, seed_num):
        # parse out sequences by seed number
        S = np.array(S[seed_num])
        print ('Startpoint',seed_num)
        cur_seq = S
        cur_like = -np.inf
        flag = False
        for opt_i in range(int(n_iter)):
            print ('EM iteration',opt_i+1)
            seq, like = self.optimise_seq(cur_seq)
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

    def fit_tebm(self, n_zscores, z_max, n_start, n_iter, n_cores, cut_controls=False):
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
        self.covars_prior = np.tile(np.identity(1), (self.n_stages, self.n_features))
        # might want to fit sequence without controls
        if cut_controls:
            print ('Cutting controls from sequence fit!')
            X, lengths = [], []
            for i in range(len(self.lengths)):
                if labels[i] != 0:
                    nobs_i = self.lengths[i]
                    for x in self.X[np.sum(self.lengths[:i]):np.sum(self.lengths[:i])+nobs_i]:
                        X.append(x)
                    lengths.append(self.lengths[i])
            self.X = np.array(X)
            self.lengths = np.array(lengths)
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
        copier = partial(self.init_seq)
        # will return shape (n_start, 1)
        seq_mat = np.array(pool.map(copier, range(n_start)))
        # now optimise
        copier = partial(self.seq_em,
                         seq_mat[:,0],
                         n_iter)
        # will return shape (n_start, 2)
        par_mat = np.array(pool.map(copier, range(n_start)))
        # distribute to local matrices
        for i in range(n_start):
            ml_seq_mat[:, :, i] = par_mat[i, 0]
            ml_like_mat[i] = par_mat[i, 1]
        ix = np.argmax(ml_like_mat)
        ml_seq = ml_seq_mat[:, :, ix]
        ml_like = ml_like_mat[ix]
        # refit model on ML sequence
        self.S = ml_seq[0]
        self.covars = self.covars_prior
        self.means = self.get_means()
        self.fit()
        return ml_seq
