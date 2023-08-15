# TEBM simulation functions
# Author: Peter Wijeratne (p.wijeratne@pm.me)
# Functions "gen_data_zscore", "gen_model_zscore", "gen_data_mixture", "gen_model_mixture" are adapted from pySuStaIn (https://github.com/ucl-pond/pySuStaIn)

import numpy as np
from scipy.stats import norm

def gen_data(n_subtypes,
             n_ppl,
             n_bms,
             n_obs,
             n_components,
             model_type='GMM',
             is_cut=False,
             n_zscores=None,
             z_max=None,
             sigma_noise=1.0,
             seq=[],
             fractions=[1],
             fwd_only=True,
             order=1,
             time_mean=1,
             verbose=False):
    # intialise z-score stuff
    if model_type=='Zscore':
        z_val_arr = np.array([[x+1 for x in range(n_zscores)]]*n_bms)
        z_max_arr = np.array([z_max]*n_bms)
        IX_vals = np.array([[x for x in range(n_bms)]*n_zscores]).T
        stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
        stage_zscore = np.array([y for x in z_val_arr.T for y in x])
        stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
        min_biomarker_zscore = [0]*n_bms
        max_biomarker_zscore = z_max_arr
    # transition generator matrix and initial probability
    # TODO: design suitable timescale and observation times for simulations
    # FIXME: also allow for fixed intervals (see HACK below)
    Q_subtypes, pi0_subtypes = [], []
    for s in range(n_subtypes):
        Q = np.zeros((n_components, n_components))
        for i in range(n_components):
            vec = np.ones(n_components-1)
            vec /= np.sum(vec)
            Q[i,:i] = vec[:i]
            Q[i,i+1:] = vec[i:]
            #            Q[i,i] = -time_mean[s]
            #            Q[i,i] = -np.random.rand()
            #            Q[i,i] = -(.25 + np.random.rand()*3.75)
            #            Q[i,i] = -(.5 + np.random.rand()*1.5)
            Q[i,i] = -(.1 + np.random.rand()*3.9)
        # zero-out forbidden states
        if fwd_only:
            for i in range(n_components):
                for j in range(n_components):
                    if j<i:
                        Q[i,j] = 0            
        if order:
            for i in range(n_components):
                for j in range(n_components):
                    if not (j<=(i+order) and j>=(i-order)):
                        Q[i,j] = 0
        # renormalise
        for i in range(n_components):
            scale = np.sum([x if jj!=i else 0 for jj,x in enumerate(Q[i])])
            for j in range(n_components):
                if i!=j:
                    if scale!=0:
                        Q[i,j] *= -Q[i,i]/scale
                    else:
                        Q[i,j] = 0.
                elif i==(n_components-1) and j==(n_components-1) and fwd_only:
                    Q[i,j] = 0.
        """
        rates = np.array([-1.*(.5+np.random.rand()*.5) for i in range(n_components)])
        Q = np.zeros((n_components, n_components))
        for i in range(n_components-1):
            temp = []
            for j in range(i):
                temp.append(0)
            temp.append(rates[i])
            temp.append(-rates[i])
            for j in range(2+i,n_components):
                temp.append(0)
            Q[i] = temp
        """
        # always set initial probability as uniform
        pi0 = np.ones(len(Q))
        Q_subtypes.append(Q)
        pi0_subtypes.append(pi0)
    # true sojourns from generated transition rate matrix
    sojourns_true = []
    for i in range(len(Q)-1):
        #        temp = [(1/Q[i,i])*np.log(np.random.rand()) for x in range(1000)]
        #        print ('True mean duration', np.mean(temp))
        sojourn_i = -1/Q[i,i]
        if verbose:
            print ('Stage',i,'true sojourn', sojourn_i)
        sojourns_true.append(sojourn_i)
    if verbose:
        print ('Total sequence true sojourn', np.sum(sojourns_true))
    # Markov sequence generation
    stages, times, jumps = [], [], []
    # total time spent in each state
    sojourn = np.zeros(n_components)
    # number of occurrences of each state
    counts = np.zeros(n_components)
    for i in range(n_ppl):
        # generate full jump process
        # tvec is time of transition, xvec is stage at corresponding time
        #        tvec, xvec, dt = sim_markov(Q_subtypes[subtypes[i]], pi0_subtypes[subtypes[i]])
        tvec, xvec, dt = sim_markov(Q_subtypes[0], pi0_subtypes[0])
        sojourn += dt
        for j in range(len(xvec)):
            counts[int(xvec[j])] += 1
        # each subsequent time step should be distributed around 1 unit of time
        # simulate 100 observation times to generate exact same dataset each run for direct comparison between models, then select number of desired observations after
        # first observation time = 0 to ensure process starts in first state
        time_steps = [0]
        # FIXME: change the range to allow irregular sampling
        for j in range(9): # can set this to whatever
            #        time_steps.append(1 + np.random.normal(scale=.05))
            #FIXME: set minmax timestep and scale externally
            #            time_steps.append(np.random.randint(1,4)*time_mean[0])
            time_steps.append(np.random.randint(1,5))
        time_i = np.cumsum(time_steps)
        times.append(time_i)
        jump_i = np.diff(time_i)
        jump_i = np.insert(jump_i, 0, 0)
        ###FIXME: HACK
        jumps.append(jump_i)
        #        print ('HACKING SIMULATED JUMPS!')
        #        jumps.append(np.array([0 if ii==0 else 1 for ii in range(len(jump_i))]))
        ###
        # sample stages corresponding to these times from the full jump process
        stages.append(step_fun(time_i, tvec, xvec))
    stages = np.array(stages)
    times = np.array(times)
    jumps = np.array(jumps)
    if is_cut:
        # for testing - cut some people at > stage_threshold at baseline
        if model_type=='Zscore':
            stage_threshold = n_components-3
        else:
            stage_threshold = n_components-1
        del_idxs = []
        for i in range(len(stages)):
            if stages[i,0] >= stage_threshold:# and np.random.rand() > .5:
                del_idxs.append(i)
        stages = np.delete(stages, del_idxs, axis=0)
        times = np.delete(times, del_idxs, axis=0)
        jumps = np.delete(jumps, del_idxs, axis=0)
        print ('##########################################################')
        print ('Cut', n_ppl-stages.shape[0], 'individuals at baseline for testing')
        print ('##########################################################')
        n_ppl = len(stages)
    # generate subtypes
    subtypes = np.random.choice(range(n_subtypes), n_ppl, replace=True, p=fractions).astype(int)
    # reduce to the desired number of observations
    if n_obs:
        stages = stages[:,:n_obs]
        times = times[:,:n_obs]
        jumps = jumps[:,:n_obs]
        lengths = np.array([n_obs for x in range(n_ppl)])
    else:
        lengths = []
        for row in stages:
            print (row.shape)
            lengths.append(row.shape[0])
        lengths = np.array(lengths)
    """
    else:
        lengths = []
        for i in range(n_ppl):
            nobs_i = np.random.randint(1,3)
            lengths.append(nobs_i)
        lengths = np.array(lengths).astype(int)
        times_temp, stages_temp, jumps_temp = [], [], []
        for i in range(n_ppl):
            stages_i = stages[i,:lengths[i]]
            times_i = times[i,:lengths[i]]
            jumps_i = jumps[i,:lengths[i]]
            stages_temp.append(stages_i)
            times_temp.append(times_i)
            jumps_temp.append(jumps_i)
        stages = np.array(stages_temp)
        times = np.array(times_temp)
        jumps = np.array(jumps_temp)
    """
    # generate data
    if model_type=='Zscore':
        if len(seq)==0:
            seq = gen_model_zscore(stage_zscore, stage_biomarker_index, n_subtypes)
        X, X_denoised = gen_data_zscore(subtypes,
                                        stages,
                                        seq,
                                        min_biomarker_zscore,
                                        max_biomarker_zscore,
                                        [sigma_noise]*n_bms,
                                        stage_zscore,
                                        stage_biomarker_index)
    else:
        if len(seq)==0:
            seq = gen_model_mixture(n_bms)
        X, X_denoised = gen_data_mixture(stages, seq, 'mixture_GMM', sigma_noise)
    # true sojourns from generated data
    for s in range(n_subtypes):
        stages_s = stages[subtypes==s]
        if verbose:
            print ('Subtype',s)
            print ('n_ppl',len(stages_s))
        sojourns_true = []
        for i in range(len(Q)-1):
            mask = stages_s[:,0]==i
            den = np.sum(mask)
            num = 0
            for j in range(1,n_obs):
                num += np.sum(stages_s[mask][:,j]!=i)
            prob_diag_i = 1-num/den
            sojourn_i = 1/(1-prob_diag_i)
            sojourns_true.append(sojourn_i)
            if verbose:
                print ('Stage',i,'true diagonal probability',round(prob_diag_i,2))
                print ('Stage',i,'true generated sojourn',round(sojourn_i,2))
        sojourns_true = np.array(sojourns_true)
        if verbose:
            print ('Total sequence true generated sojourn',round(np.nansum(sojourns_true[~np.isinf(sojourns_true)]),2))
    # get data in long format for TEBM
    X0 = []
    stages_0 = []
    for i in range(n_ppl):
        X0.append(X[i][:,0])
        stages_0.append(stages[i][0])
    X0 = np.array(X0)
    stages_0 = np.array(stages_0)
    X_temp, stages_temp, times_temp, jumps_temp = [], [], [], []
    for i in range(n_ppl):
        X_i = X[i]
        for j in range(X_i.shape[1]):
            X_temp.append(X_i[:,j])
        stage_i = stages[i]
        for j in range(stage_i.shape[0]):
            stages_temp.append(stage_i[j])
        time_i = times[i]
        for j in range(time_i.shape[0]):
            times_temp.append(time_i[j])
        jump_i = jumps[i]
        for j in range(jump_i.shape[0]):
            jumps_temp.append(jump_i[j])
    X = np.array(X_temp)
    stages = np.array(stages_temp)
    times = np.array(times_temp)
    jumps = np.array(jumps_temp)
    # choose which subjects will be cases and which will be controls
    MIN_CASE_STAGE = np.round((n_bms + 1) * 0.8)
    index_case = np.where(stages_0 >=  MIN_CASE_STAGE)[0]
    index_control = np.where(stages_0 ==  0)[0]
    labels = 2 * np.ones(n_ppl, dtype=int) # 2 - intermediate value, not used in mixture model fitting
    labels[index_case] = 1 # 1 - cases
    labels[index_control] = 0 # 0 - controls
    return X, lengths, jumps, labels, X0, stages, times, seq, Q_subtypes, pi0_subtypes, subtypes

def sim_markov(Q,
               pi0,
               n_jumps=None):
    """
    Gillespie's direct stochastic simulation algorithm for a single Markov chain with absorbing final state
    """
    n_s = len(pi0)
    xvec = np.zeros(n_s)
    tvec = np.zeros(n_s)
    x = np.random.choice(n_s, size=1, p=pi0/np.sum(pi0))[0]
    t = 0
    xvec[0] = x
    tvec[0] = 0.
    if not n_jumps:
        n_jumps = n_s-1
    for i in range(n_jumps):
        # final state is absorbing - don't increment time
        if Q[x,x] != 0:
            t += (1/Q[x,x])*np.log(np.random.random())
        weights = Q[x].copy()
        weights[x] = 0
        # final state is absorbing - don't change state
        if np.sum(weights) != 0:
            x = np.random.choice(n_s, size=1, p=weights/np.sum(weights))[0]
        xvec[i+1] = x
        tvec[i+1] = t
    # time spent in each state
    dt = np.zeros(len(pi0))
    for i in range(len(tvec)):
        if i < (len(tvec)-1):
            dt[int(xvec[i])] += tvec[i+1] - tvec[i]
    return tvec, xvec, dt

def step_fun(x, xvec, yvec):
    y = []
    for i in range(len(x)):
        for j in range(len(xvec)-1):
            if x[i] >= xvec[j] and x[i] < xvec[j+1]:
                y.append(yvec[j])
            elif x[i] >= xvec[-1]:                
                y.append(yvec[-1])
                break
    return y

def gen_data_zscore(subtypes,
                    stages,
                    gt_ordering,
                    min_biomarker_zscore,
                    max_biomarker_zscore,
                    std_biomarker_zscore,
                    stage_zscore,
                    stage_biomarker_index):

    N = stage_biomarker_index.shape[1]
    N_S = gt_ordering.shape[0]    
    possible_biomarkers = np.unique(stage_biomarker_index)
    B = len(possible_biomarkers)
    stage_value = np.zeros((B,N+2,N_S))    
    for s in range(N_S):
        S = gt_ordering[s,:]
        S_inv = np.array([0]*N)
        S_inv[S.astype(int)] = np.arange(N)
        for i in range(B):
            b = possible_biomarkers[i]
            event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
            event_value = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])
            for j in range(len(event_location)-1):
                if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    index = np.arange(event_location[j],event_location[j+1]+2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                else:
                    index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)
    stage_value = 0.5 * stage_value[:, :stage_value.shape[1] - 1, :] + 0.5 * stage_value[:, 1:, :]
    M = stages.shape[0]
    # initialise variable observation length arrays
    data = []
    for i in range(len(stages)):
        data.append(np.zeros((B, len(stages[i]))))
    data_denoised = []
    for i in range(len(stages)):
        data_denoised.append(np.zeros((B, len(stages[i]))))
    # set data
    for i in range(M):
        stage_i = stages[i]
        # assume noise homoskedastic
        noise = np.random.normal(np.zeros(B), std_biomarker_zscore, B)
        for t in range(len(stage_i)):
            for j in range(B):
                data_denoised[i][j][t] = stage_value[:,int(stage_i[t]),subtypes[i]][j] # last index would be "subtypes[i]"
                data[i][j][t] = data_denoised[i][j][t] + noise[j]
    return data, data_denoised

def gen_model_zscore(stage_zscore,
                     stage_biomarker_index,
                     N_S):

    N = np.array(stage_zscore).shape[1]
    S = np.zeros((N_S,N))
    for s in range(N_S):
        for i in range(N):
            IS_min_stage_zscore = np.array([False]*N)
            possible_biomarkers = np.unique(stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected = [False]*N
                for k in set(range(N))-set(S[s][:i]):
                    IS_unselected[k] = True
                this_biomarkers = np.array([(np.array(stage_biomarker_index)[0]==possible_biomarkers[j]).astype(int)+(np.array(IS_unselected)==1).astype(int)])==2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore = 0
                else:
                    this_min_stage_zscore = min(stage_zscore[this_biomarkers])
                if(this_min_stage_zscore):
                    temp = ((this_biomarkers.astype(int)+(stage_zscore==this_min_stage_zscore).astype(int))==2).T
                    temp = temp.reshape(len(temp),)
                    IS_min_stage_zscore[temp]=True
            events = np.array(range(N))
            possible_events = np.array(events[IS_min_stage_zscore])
            this_index = np.ceil(np.random.rand()*((len(possible_events))))-1
            S[s][i] = possible_events[int(this_index)]
    return S

def gen_model_mixture(N_biomarkers):    
    return np.array([np.random.permutation(N_biomarkers)]).astype(float)

def gen_data_mixture(stages,
                     gt_ordering,
                     mixture_style,
                     sigma_noise=1.):
    N_biomarkers                        = gt_ordering.shape[1]
    N_subjects                          = len(stages)
    #controls are always drawn from N(0, 1) distribution
    mean_controls                       = np.array([0]   * N_biomarkers)
    std_controls                        = np.array([sigma_noise] * N_biomarkers)
    #mean and variance for cases
    #if using mixture_GMM, use normal distribution with mean 1 and std. devs sampled from a range
    if mixture_style == 'mixture_GMM':
        #        mean_cases                       = np.array(np.random.uniform(size=N_biomarkers)+1.35) # PW: 1.5 to look more like ADNI SNR
        mean_cases                       = np.array(np.random.uniform(size=N_biomarkers)+1.5) # PW: 1.5 to look more like ADNI SNR
        std_cases                        = np.array([sigma_noise] * N_biomarkers)
    #if using mixture_KDE, use log normal with mean 0.5 and std devs sampled from a range
    elif mixture_style == 'mixture_KDE':
        mean_cases                      = np.array([0.5] * N_biomarkers)
        std_cases                       = np.random.uniform(0.2, 0.5, N_biomarkers)
    # initialise variable observation length arrays
    data = []
    for i in range(len(stages)):
        data.append(np.zeros((N_biomarkers, len(stages[i]))))
    data_denoised = []
    for i in range(len(stages)):
        data_denoised.append(np.zeros((N_biomarkers, len(stages[i]))))
    #loop over all subjects, creating measurment for each biomarker based on what subtype and stage they're in
    for i in range(N_subjects):
        stage_i = stages[i]
        for t in range(len(stage_i)):
            S_i                               = gt_ordering[0, :].astype(int) # first index would be subtype
            stage_i_t                         = stage_i[t].astype(int)
            #fill in with ABNORMAL values up to the subject's stage
            for j in range(stage_i_t):
                if      mixture_style == 'mixture_KDE':
                    sample_j                = np.random.lognormal(mean_cases[S_i[j]], std_cases[S_i[j]])
                elif    mixture_style == 'mixture_GMM':
                    sample_j                = np.random.normal(mean_cases[S_i[j]], std_cases[S_i[j]])
                data[i][S_i[j]][t]             = sample_j
                data_denoised[i][S_i[j]][t]    = mean_cases[S_i[j]]
            # fill in with NORMAL values from the subject's stage+1 to last stage
            for j in range(stage_i_t, N_biomarkers):
                data[i][S_i[j]][t]             = np.random.normal(mean_controls[S_i[j]], std_controls[S_i[j]])
                data_denoised[i][S_i[j]][t]    = mean_controls[S_i[j]]
    return data, data_denoised
