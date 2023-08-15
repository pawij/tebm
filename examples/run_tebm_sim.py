# Run TEBM simulation
# Author: Peter Wijeratne (p.wijeratne@pm.me)

import sys
from tebm import tebm_fix, tebm_var
from kde_ebm import plotting
from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat

import numpy as np
import pandas as pd
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import multiprocessing

from sim_funcs import gen_data
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

############################################################################################################## USER INPUT START
is_ebm = False # set True if you want to use the standard EBM
is_cut = False # cut people starting in final state(s); for testing effect of right-censoring
sigma_noise = 0.1 # simulated measurement noise
n_ppl = 100 # number of people
n_bms = 5 # number of biomarkers
n_obs = 2 # number of observations per person. Set = None for random number of observations per person
model_type = 'GMM' # type of likelihood model, can be one of: 'GMM', KDE', 'Zscore'
if model_type == 'Zscore':
    n_zscores = 3 # number of z-score events per biomarker
    z_max = 5 # maximum z-score event per biomarker
    order = 1#int(n_bms*n_zscores) # transition order. For unconstrained matrix, set order = number of events-1
    n_components = int(n_bms*n_zscores + 1)
else:
    order = 1 # transition order. For unconstrained matrix, set order = n_bms
    n_components = n_bms + 1
fwd_only = True # set True to only allow forward transitions
############################################################################################################## USER INPUT END
scale = 1 # mean time scale for each event 
time_mean = scale # used when simulating data
init_params = 's' # initialise start / initial probability to uniform prior
fit_params = 'st' # fit start probability and transition matrix
tol = 1E-3 # tolerance for both inner and outer EM
n_cores = 4 # number of cores used for parallelising start point
n_start = 4 # number of outer EM start points 
# set random seed
try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
np.random.seed(seed)
if model_type=='Zscore':
    n_stages = int(n_zscores*n_bms + 1)
else:
    n_stages = n_bms + 1
algo = 'viterbi' # staging algorithm
n_iter_outer = 100 # maximum number of outer EM iterations (for fitting sequence)
# if fitting standard EBM then we don't need the inner EM loop (for fitting start probability and transition matrix)
if is_ebm:
    n_iter_inner = 0
else:
    n_iter_inner = 1
biom_labels = []
if model_type=='Zscore':
    for i in range(n_bms):
        for j in range(n_zscores):
            biom_labels.append('BM'+str(i)+'Z'+str(j))
else:
    for i in range(n_bms):
        biom_labels.append('BM'+str(i))
plot_raw_data = False
# generate data
if model_type=='Zscore':
#    X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0 = gen_data(n_ppl, n_bms, n_obs, n_stages, model_type=model_type, is_cut=is_cut, n_zscores=n_zscores, z_max=z_max, sigma_noise=sigma_noise)
    X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(1, n_ppl, n_bms, n_obs, n_components, model_type=model_type, is_cut=is_cut, n_zscores=n_zscores, z_max=z_max, sigma_noise=sigma_noise, seq=[], fractions=[1], fwd_only=fwd_only, order=order, time_mean=[1/scale])
else:
    X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(1, n_ppl, n_bms, n_obs, n_components, model_type=model_type, is_cut=is_cut, n_zscores=None, z_max=None, sigma_noise=sigma_noise, seq=[], fractions=[1], fwd_only=fwd_only, order=order, time_mean=[1/scale])

save_variables = {}
save_variables["X"] = X
save_variables["lengths"] = lengths
save_variables["jumps"] = jumps
save_variables["labels"] = labels
save_variables["X0"] = X0
save_variables["seq_true"] = seq_true
save_variables["times"] = times
save_variables["Q"] = Q
save_variables["pi0"] = pi0
pickle_file = open('./simdata_Nppl'+str(n_ppl)+'_Nbms'+str(n_bms)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'.pickle', 'wb')
pickle_output = pickle.dump(save_variables, pickle_file)
pickle_file.close()

"""
pickle_file = open('simdata_Nppl'+str(n_ppl)+'_Nbms'+str(n_bms)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'.pickle', 'rb')
loaded_variables = pickle.load(pickle_file)
X = loaded_variables["X"]
lengths = loaded_variables["lengths"]
jumps = loaded_variables["jumps"]
labels = loaded_variables["labels"]
X0 = loaded_variables["X0"]
seq_true = loaded_variables["seq_true"]
times = loaded_variables["times"]
Q = loaded_variables["Q"]
pi0 = loaded_variables["pi0"]
""" 
# EBM treats repeated measurements as from separate individuals
if is_ebm:
    labels0 = labels.copy()
    labels_temp = []
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        for j in range(nobs_i):
            labels_temp.append(labels[i])
    labels = np.array(labels_temp)        
    lengths = np.ones(X.shape[0]).astype(int)
        
    print (X.shape, lengths.shape)
#

labels_long = []
for i in range(len(lengths)):
    nobs_i = lengths[i]
    for j in range(nobs_i):
        labels_long.append(labels[i])
labels_long = np.array(labels_long)
snr = np.nanmean(X[labels_long!=0], axis=0)/np.nanstd(X[labels_long==0], axis=0)
#for i in range(len(biom_labels)):
#    print (biom_labels[i]+' SNR =', snr[i])
print ('SNR', snr)

# this is currently redundant, but not for long...
obs_type = 'Var'
#
print ('n_ppl', X0.shape[0], 'n_bms', X0.shape[1], 'n_obs', X.shape[0], 'n_intervals', len(np.unique(jumps))-1, 'n_stages', n_stages, 'order', order, 'fwd_only', fwd_only)

print ('Fitting '+model_type+'-'+obs_type+'-TEBM...')
if model_type == 'GMM' or model_type == 'KDE':
    if obs_type == 'Fix':
        model = tebm_fix.MixtureTEBM(X=X,
                                     lengths=lengths, 
                                     n_stages=n_stages,
                                     time_mean=time_mean,
                                     n_iter=n_iter_inner,
                                     fwd_only=fwd_only,
                                     order=order,
                                     algo=algo)
        seq_model, mixtures = model.fit_tebm(labels, n_start=n_start, n_iter=n_iter_outer, n_cores=n_cores, model_type=model_type, cut_controls=False)
    else:
        model = tebm_var.MixtureTEBM(X=X, lengths=lengths, jumps=jumps,
                                     n_components=n_components, time_mean=time_mean, covariance_type="diag",
                                     n_iter=n_iter_inner, tol=tol,
                                     init_params=init_params, params=fit_params,
                                     algorithm=algo, verbose=False, allow_nan=True,
                                     fwd_only=fwd_only, order=order)
        seq_model, mixtures = model._fit_tebm(labels, n_start=n_start, n_iter=n_iter_outer, n_cores=n_cores, model_type=model_type, cut_controls=False)
elif model_type == 'Zscore':
    model = tebm_fix.ZscoreTEBM(X=X,
                                lengths=lengths, 
                                n_stages=n_stages,
                                time_mean=1/n_stages,
                                n_iter=n_iter_inner,
                                fwd_only=fwd_only,
                                order=order,
                                algo=algo)
    seq_model = model.fit_tebm(n_zscores=n_zscores, z_max=z_max, n_start=n_start, n_iter=n_iter_outer, n_cores=n_cores, cut_controls=False)
else:
    print ('Likelihood model not recognised! quit()')
    quit()

fig, ax = plotting.mixture_model_grid(X0, labels, mixtures, biom_labels)
for i in range(len(ax)):
    for j in range(len(ax)-1):
        ax[i,j].set_yscale('log')

print ('True seq',seq_true[0])
print ('MaxL seq',seq_model[0])
print ('Kendall tau',sp.stats.kendalltau(seq_true[0], seq_model[0]))

if not is_ebm:
    n_iter_inner = 100

# refit with 100 iterations
if obs_type == 'Var':
    model = tebm_var.MixtureTEBM(X=X, lengths=lengths, jumps=jumps,
                                 n_components=n_components, time_mean=time_mean, covariance_type="diag",
                                 n_iter=n_iter_inner, tol=tol,
                                 init_params=init_params, params=fit_params,
                                 algorithm=algo, verbose=True, allow_nan=True,
                                 fwd_only=fwd_only, order=order)
else:
    model = tebm_fix.MixtureTEBM(X=X,
                                 lengths=lengths, 
                                 n_stages=n_stages,
                                 time_mean=time_mean,
                                 n_iter=n_iter_inner,
                                 fwd_only=fwd_only,
                                 order=order,
                                 algo=algo,
                                 verbose=True)
model.S = seq_model[0]
if model_type=='GMM':
    model.prob_mat = get_prob_mat(X, mixtures)
    model.mixtures = mixtures
    model.fit()

if is_ebm:
    fout_name = 'simrun'+str(seed)+'_'+model_type+'-ebm_Nppl'+str(n_ppl)+'_Nstates'+str(n_stages)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'_Nstart'+str(n_start)+'_iscut_'+str(is_cut)+'_noise_'+str(sigma_noise)[0]+'p'+str(sigma_noise)[2]+'_order_'+str(order)+'_fwdonly_'+str(fwd_only)+'_Nits_'+str(n_iter_inner)+'.pickle'
else:
    fout_name = 'simrun'+str(seed)+'_'+model_type+'-tebm_Nppl'+str(n_ppl)+'_Nstates'+str(n_stages)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'_Nstart'+str(n_start)+'_iscut_'+str(is_cut)+'_noise_'+str(sigma_noise)[0]+'p'+str(sigma_noise)[2]+'_order_'+str(order)+'_fwdonly_'+str(fwd_only)+'_Nits_'+str(n_iter_inner)+'.pickle'
save_variables = {}
save_variables["X"] = X
save_variables["lengths"] = lengths
save_variables["jumps"] = jumps
save_variables["labels"] = labels
save_variables["seq_true"] = seq_true
save_variables["stages_true"] = stages_true
save_variables["Q_true"] = Q[0]
save_variables["p_vec_true"] = pi0
save_variables["seq_model"] = seq_model
save_variables["Q_model"] = model.Q_
save_variables["p_vec_model"] = model.startprob_

pickle_file = open('./'+fout_name, 'wb')
pickle_output = pickle.dump(save_variables, pickle_file)
pickle_file.close()

if plot_raw_data:
    n_x = np.round(np.sqrt(n_bms)).astype(int)
    n_y = np.ceil(np.sqrt(n_bms)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
    for i in range(n_bms):
        for j in range(len(lengths)):
            nobs_i = lengths[j]
            s_idx, e_idx = int(np.sum(lengths[:j])), int(np.sum(lengths[:j])+nobs_i)
            ax[i // n_x, i % n_x].plot(stages_true[s_idx:e_idx],X[s_idx:e_idx,i])
            ax[i // n_x, i % n_x].scatter(stages_true[s_idx:e_idx],X[s_idx:e_idx,i])
            ax[i // n_x, i % n_x].set_title(biom_labels[i])
            
# plots
if plot_raw_data:
    if model_type == 'GMM' or model_type == 'KDE':
        # biomarker distributions and mixture model fits
        fig, ax = plotting.mixture_model_grid(X0, labels, mixtures, biom_labels)

# true transition matrix
transmat = np.zeros((n_stages,n_stages))
startprob = np.zeros(n_stages)
for i in range(len(lengths)):
    nobs_i = lengths[i]
    s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+nobs_i)
    stages_i = stages_true[s_idx:e_idx].astype(int)
    startprob[stages_i[0]] += 1
    for j in range(1,len(stages_i)):
        transmat[stages_i[j-1],stages_i[j]] += 1
# normalise across rows
startprob /= np.sum(startprob)
for i in range(transmat.shape[0]):
    transmat[i] /= np.sum(transmat[i])
# plot true initial probability
fig, ax = plt.subplots()
ax.bar(np.arange(n_stages),startprob)
ax.set_xlabel('Stage', fontsize=18, labelpad=8)
ax.set_ylabel('Probability', fontsize=18, labelpad=2)
ax.tick_params(labelsize=18)
plt.subplots_adjust(top=0.95, right=0.99, bottom=.15)
ax.set_title('True pi0')
# initial probability
if obs_type == 'Fix':
    startprob = model.p_vec
else:
    startprob = model.startprob_    
fig, ax = plt.subplots()
ax.bar(np.arange(n_stages),startprob)
ax.set_xlabel('Stage', fontsize=18, labelpad=8)
ax.set_ylabel('Probability', fontsize=18, labelpad=2)
ax.tick_params(labelsize=18)
plt.subplots_adjust(top=0.95, right=0.99, bottom=.15)
ax.set_title('Fitted pi0')
# sojourn times
sojourns_reco, sojourns_true = [], []
sojourns_reco.append(0)
sojourns_true.append(0)
for i in range(len(transmat)-1):     # skip final stage (absorbing)
    if obs_type == 'Fix':
        sojourns_reco.append(1/(1-transmat[i,i])/scale)
    else:
        sojourns_reco.append(-1/model.Q_[i,i]/scale)
        sojourns_true.append(-1/Q[0][i,i]/scale)
#    print ('Stage',i,'duration',round(sojourns_reco[i+1],2))
sojourns_reco = np.array(sojourns_reco)
sojourns_true = np.array(sojourns_true)
print ('np.sum(sojourns_reco)',round(np.sum(sojourns_reco),2))
print ('np.sum(sojourns_true)',round(np.sum(sojourns_true),2))
print ('np.abs(np.sum(sojourns_reco)-np.sum(sojourns_true))',np.abs(np.sum(sojourns_reco)-np.sum(sojourns_true)))
print ('np.sqrt(np.sum(np.power(sojourns_reco-sojourns_true, 2))/len(sojourns))',np.sqrt(np.sum(np.power(sojourns_reco-sojourns_true, 2))/len(sojourns_reco)))
# staging
if obs_type == 'Fix':
    stages_model = model.stages(X, lengths)
else:
    stages_model, _ = model.predict(X, lengths, jumps)
stages_true = stages_true.flatten()
scale = [10.]*len(stages_true)
for i in range(len(stages_true)):
    x0 = stages_true[i]
    x1 = stages_model[i]
    for j in range(len(stages_true)):
        if x0 == stages_true[j] and x1 == stages_model[j]:
            scale[i] += 20.
fig, ax = plt.subplots()
ax.scatter(stages_true.flatten(), stages_model, s=scale)
ax.set_xlabel('Stage (true)')
ax.set_ylabel('Stage (reco)')
ax.grid()

if obs_type == 'Var':
    # true transition rate matrix
    transmat = np.real(Q[0])
    fig, ax = plt.subplots()
    ax.imshow(transmat, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            if abs(round(transmat[i, j], 3)) > 1E-3:
                text = ax.text(j, i, round(transmat[i, j], 3), ha="center", va="center", color="black", size=10)
    event_labels = np.array(biom_labels)[seq_true[0].astype(int)]
    event_labels = np.insert(event_labels, 0, 'None')
    ax.set_xticks(np.arange(len(event_labels)))
    ax.set_yticks(np.arange(len(event_labels)))
    xticklabels = []
    for x in event_labels:
        xticklabels.append(str(x)+' (t1)')
    ax.set_xticklabels(xticklabels, ha='right', rotation=45, rotation_mode='anchor', fontsize=12)
    yticklabels = []
    for x in event_labels:
        yticklabels.append(str(x)+' (t0)')
    ax.set_yticklabels(yticklabels, ha='right', rotation_mode='anchor', fontsize=12)
    plt.subplots_adjust(bottom=.2, top=.95)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title('True Q')
    # prior transition rate matrix
    transmat = np.real(model.Q_prior)
    fig, ax = plt.subplots()
    ax.imshow(transmat, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            if abs(round(transmat[i, j], 3)) > 1E-3:
                text = ax.text(j, i, round(transmat[i, j], 3), ha="center", va="center", color="black", size=10)
    event_labels = np.array(biom_labels)[seq_true[0].astype(int)]
    event_labels = np.insert(event_labels, 0, 'None')
    ax.set_xticks(np.arange(len(event_labels)))
    ax.set_yticks(np.arange(len(event_labels)))
    xticklabels = []
    for x in event_labels:
        xticklabels.append(str(x)+' (t1)')
    ax.set_xticklabels(xticklabels, ha='right', rotation=45, rotation_mode='anchor', fontsize=12)
    yticklabels = []
    for x in event_labels:
        yticklabels.append(str(x)+' (t0)')
    ax.set_yticklabels(yticklabels, ha='right', rotation_mode='anchor', fontsize=12)
    plt.subplots_adjust(bottom=.2, top=.95)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title('Prior Q')
    # fitted transition rate matrix
    transmat = np.real(model.Q_)
    fig, ax = plt.subplots()
    ax.imshow(transmat, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            if abs(round(transmat[i, j], 3)) > 1E-3:
                text = ax.text(j, i, round(transmat[i, j], 3), ha="center", va="center", color="black", size=10)
    event_labels = np.array(biom_labels)[seq_model[0].astype(int)]
    event_labels = np.insert(event_labels, 0, 'None')
    ax.set_xticks(np.arange(len(event_labels)))
    ax.set_yticks(np.arange(len(event_labels)))
    xticklabels = []
    for x in event_labels:
        xticklabels.append(str(x)+' (t1)')
    ax.set_xticklabels(xticklabels, ha='right', rotation=45, rotation_mode='anchor', fontsize=12)
    yticklabels = []
    for x in event_labels:
        yticklabels.append(str(x)+' (t0)')
    ax.set_yticklabels(yticklabels, ha='right', rotation_mode='anchor', fontsize=12)
    plt.subplots_adjust(bottom=.2, top=.95)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title('Fitted Q')
    #    print ('sum(diag(Q_true)-diag(Q_reco))', np.abs(np.sum(np.abs(np.diag(Q[0]))-np.abs(np.diag(model.Q_)))))
    #    print ('sum(diag(Q_true)-diag(Q_reco))', np.sum(np.abs(np.diag(Q[0])-np.diag(model.Q_))))
    print ('sum(diag(Q_true)-diag(Q_reco))', np.sqrt(np.sum(np.power(np.diag(Q[0])-np.diag(model.Q_), 2))/model.Q_.shape[0]))

# write
# write data
if is_ebm:
    fout_name = 'simrun'+str(seed)+'_'+model_type+'-ebm_Nppl'+str(n_ppl)+'_Nstates'+str(n_stages)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'_Nstart'+str(n_start)+'_iscut_'+str(is_cut)+'_noise_'+str(sigma_noise)[0]+'p'+str(sigma_noise)[2]+'_order_'+str(order)+'_fwdonly_'+str(fwd_only)+'_Nits_'+str(n_iter_inner)+'.pickle'
else:
    fout_name = 'simrun'+str(seed)+'_'+model_type+'-tebm_Nppl'+str(n_ppl)+'_Nstates'+str(n_stages)+'_Nobs'+str(n_obs)+'_Nintervals'+str(len(np.unique(jumps))-1)+'_Nstart'+str(n_start)+'_iscut_'+str(is_cut)+'_noise_'+str(sigma_noise)[0]+'p'+str(sigma_noise)[2]+'_order_'+str(order)+'_fwdonly_'+str(fwd_only)+'_Nits_'+str(n_iter_inner)+'.pickle'
save_variables = {}
save_variables["X"] = X
save_variables["lengths"] = lengths
save_variables["jumps"] = jumps
save_variables["labels"] = labels
save_variables["seq_true"] = seq_true
save_variables["stages_true"] = stages_true
save_variables["Q_true"] = Q[0]
save_variables["p_vec_true"] = pi0
save_variables["seq_model"] = seq_model
save_variables["stages_model"] = stages_model
save_variables["Q_model"] = model.Q_
save_variables["p_vec_model"] = startprob

pickle_file = open('./'+fout_name, 'wb')
pickle_output = pickle.dump(save_variables, pickle_file)
pickle_file.close()

plt.show()
