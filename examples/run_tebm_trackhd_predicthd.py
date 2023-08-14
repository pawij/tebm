# Run TEBM simulation
# Author: Peter Wijeratne (p.wijeratne@pm.me)

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import bootstrap
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sklearn
from sklearn.utils import resample
from tebm import tebm_var, tebm_fix
from kde_ebm import plotting
from kde_ebm.mixture_model import get_prob_mat
from utils import get_transmat_prior, plot_timeline, plot_examples, tebm_preds, crossval_new
import warnings
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

colors = ['C{}'.format(x) for x in range(10)]
colors.append('black')
colors.append('grey')

### USER INPUT START
data_in = 'predicthd_merged_adjusted.csv'
data_out = 'tebm_predicthd_adjusted.pickle'
biom_labels = ['Caudate','Putamen','TMS','SDMT','TFC']
plot_labels = ['Caudate','Putamen','TMS','SDMT','TFC'] # as above, but for plotting
#plot_labels = ['Biomarker 1','Biomarker 2','Biomarker 3','Biomarker 4','Biomarker 5'] # as above, but for plotting
order = 1 #len(biom_labels) # transition order. order = len(biom_labels) gives an unconstrained transition matrix
fwd_only = True # enforce forward-only transitions
model_type = 'GMM' # type of likelihood model, can be one of: 'GMM', KDE', 'Zscore'
if model_type == 'Zscore':
    n_zscores = 3 # number of z-score events per biomarker
    z_max = 5 # maximum z-score event per biomarker
baseline_only = False # might choose to run on only baseline data
cut_nans = False # drop all rows in dataset with any missing values
scale = 1
### USER INPUT END
# other buttons
np.random.seed(42)
algorithm = 'viterbi'
n_iter_inner_em = 100
tol = 1E-3
init_params = 's'
fit_params = 'st'
n_start = 4
n_iter_seq_em = 100
n_cores = 4
plot_all_data = True
# get data
data = pd.read_csv(data_in)

if baseline_only:
    print ('Using only baseline data! There will be no time dimension')
    data = data[data['Time']==0]
if cut_nans:
    data.dropna(inplace=True)
##
"""
data = data[data['Time']==0]
print (data.columns)
print (np.unique(data['DX_bl'].values, return_counts=True))
for dx in np.unique(data['DX_bl'].values):
    print ('group',dx)
    print ('age',np.nanmean(data[data['DX_bl']==dx]['AGE'].values),np.nanstd(data[data['DX_bl']==dx]['AGE'].values))
    print ('sex',np.unique(data[data['DX_bl']==dx]['PTGENDER'].values, return_counts=True))
quit()
"""
##
unique_ids = np.unique(data['RID'])
X_predict, lengths_predict, jumps_predict, labels_predict, ids, times_predict, dscore, dx_change, age, cag, subgroup, tfc, sex = [], [], [], [], [], [], [], [], [], [], [], [], []
ids_temp, labels_temp, dx_change_temp = [], [], []
labels_long = []
for i in range(len(unique_ids)):
    X_i = data.loc[data['RID']==unique_ids[i]]
    if X_i['Caudate'].values[0] > 0.001:
        continue
    
    ids.append(unique_ids[i])

#    if X_i['Thalamus Proper'].values[0] < -0.0015:
#        print ('cut!')
#        continue
    
    ### HACK - group TRACK time points into visits
    #    time_i = X_i['Time']
    time_i = X_i['Time']
    # sort by time
    idx_sort = np.argsort(time_i)
    X_i = X_i.iloc[idx_sort]
    time_i = time_i.iloc[idx_sort]
    # parse out data types
    label_i = X_i['group'].values[0]
    age_i = X_i['age'].values[0]
    sex_i = X_i['sex'].values[0]
    cag_i = X_i['cag'].values[0]
    tfc_i = X_i['TFC'].values
    X_i = X_i[biom_labels].values
    time_i = time_i.values
    # calculate intervals
    delta_time_i = np.array([x for x in np.diff(time_i)])
    delta_time_i = np.insert(delta_time_i, 0, 0)
    seen = []
    # add to arrays
    for t in range(X_i.shape[0]):
        #        t_seg = np.floor(delta_time_i[t]*4)/4
        t_seg = int(delta_time_i[t])
        if not t_seg in seen:
            X_predict.append(X_i[t])
            jumps_predict.append(t_seg)
            times_predict.append(time_i[t])
            tfc.append(tfc_i[t])
            labels_long.append(label_i)
            seen.append(t_seg)
    lengths_predict.append(len(seen))
    ###HACK
    #    if label_i == 2:
    #        label_i = 1
    labels_predict.append(label_i)
    age.append(age_i)
    sex.append(sex_i)
    cag.append(cag_i)    
X_predict = np.array(X_predict)
lengths_predict = np.array(lengths_predict)
jumps_predict = np.array(jumps_predict)
labels_predict = np.array(labels_predict)
age = np.array(age)
sex = np.array(sex)
cag = np.array(cag)
times_predict = np.array(times_predict)
tfc = np.array(tfc)
# relabel biomarker labels
biom_labels = np.array(plot_labels)
# relabel -ve labels for KDE code
labels_predict[labels_predict<0] = 3
times_predict = np.array(times_predict)
X0_predict = []
for i in range(len(lengths_predict)):
    X0_predict.append(X_predict[np.sum(lengths_predict[:i])])
X0_predict = np.array(X0_predict)
n_bms = X_predict.shape[1]
if model_type=='Zscore':
    n_components = int(n_zscores*n_bms + 1)
else:
    n_components = n_bms + 1
# get transition matrix prior
transmat_prior = get_transmat_prior(n_components, order=order, fwd_only=fwd_only)
if baseline_only:
    obs_type = 'Fix'
else:
    obs_type = 'Var'
#
print (np.mean(age[labels_predict==0]), np.std(age[labels_predict==0]))
print (np.mean(age[labels_predict==2]), np.std(age[labels_predict==2]))
print (np.mean(age[labels_predict==1]), np.std(age[labels_predict==1]))
print (np.mean(age[labels_predict==-1]), np.std(age[labels_predict==-1]))

print (np.sum(sex[labels_predict==0]==0), np.sum(sex[labels_predict==0]==1))
print (np.sum(sex[labels_predict==2]==0), np.sum(sex[labels_predict==2]==1))
print (np.sum(sex[labels_predict==1]==0), np.sum(sex[labels_predict==1]==1))
print (np.sum(sex[labels_predict==-1]==0), np.sum(sex[labels_predict==-1]==1))
#quit()
########################################################################################################HACK
#obs_type = 'Fix'
#print ('HACK! obs_type')
########################################################################################################HACK
print ('n_ppl', X0_predict.shape[0], 'n_bms', X0_predict.shape[1], 'n_obs', X_predict.shape[0], 'n_intervals', len(np.unique(jumps_predict))-1, 'n_stages', n_components, 'order', order, 'fwd_only', fwd_only)
print ('Fraction missing data', round(np.count_nonzero(np.isnan(X_predict))/(X_predict.shape[0]*X_predict.shape[1]),3))
print (np.unique(labels_predict, return_counts=True), np.max(lengths_predict), np.unique(jumps_predict, return_counts=True), np.max(times_predict))
labels_long = []
for i in range(len(lengths_predict)):
    nobs_i = lengths_predict[i]
    for j in range(nobs_i):
        labels_long.append(labels_predict[i])
labels_long = np.array(labels_long)
snr = np.nanmean(X_predict[labels_long!=0], axis=0)/np.nanstd(X_predict[labels_long==0], axis=0)
for i in range(len(biom_labels)):
    print (biom_labels[i]+' SNR =', snr[i])

if plot_all_data:
    n_x = np.round(np.sqrt(n_bms)).astype(int)
    n_y = np.ceil(np.sqrt(n_bms)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
    for i in range(n_bms):
        for j in range(len(lengths_predict)):
            nobs_i = lengths_predict[j]
            s_idx, e_idx = int(np.sum(lengths_predict[:j])), int(np.sum(lengths_predict[:j])+nobs_i)
            y = X_predict[s_idx:e_idx,i]
            x = times_predict[s_idx:e_idx][~np.isnan(y)]
            y = y[~np.isnan(y)]
            ax[i // n_x, i % n_x].plot(x, y)
            ax[i // n_x, i % n_x].set_title(biom_labels[i])
#    plt.show()

## use TRACK-HD for mixture models
data_track = pd.read_csv('trackhd_all_adjusted_units12months_icvcorrected_xsecreg.csv')
unique_ids_track = np.unique(data_track['RID'])
X_track, jumps_track, lengths_track, labels_track, ages_track = [], [], [], [], []
labels_track_long = []
for i in range(len(unique_ids_track)):
    X_i = data_track.loc[data_track['RID']==unique_ids_track[i]]
    time_i = X_i['TIME']
    # sort by time
    idx_sort = np.argsort(time_i)
    X_i = X_i.iloc[idx_sort]
    time_i = time_i.iloc[idx_sort].values
    # parse out data types
    label_i = X_i['group'].values[0]
    age_i = X_i['age'].values[0]
    X_i = X_i[['Caudate','Putamen','motorscore','sdmt','tfc']].values
    # calculate intervals
    delta_time_i = np.array([x for x in np.diff(time_i)])
    delta_time_i = np.insert(delta_time_i, 0, 0)
    # add to arrays
    lengths_track.append(X_i.shape[0])
    labels_track.append(label_i)
    for t in range(X_i.shape[0]):
        X_track.append(X_i[t])
        jumps_track.append(delta_time_i[t])
        labels_track_long.append(label_i)
    ages_track.append(age_i)
X_track = np.array(X_track)
jumps_track = np.array(jumps_track)
lengths_track = np.array(lengths_track)
labels_track = np.array(labels_track)
ages_track = np.array(ages_track)

X0_track = []
for i in range(len(lengths_track)):
    X0_track.append(X_track[np.sum(lengths_track[:i])])
X0_track = np.array(X0_track)
print (sp.stats.ttest_ind(X0_track[labels_track==0][:,3],X0_track[labels_track==1][:,3],nan_policy='omit'))

from kde_ebm.mixture_model import fit_all_gmm_models
mixtures = fit_all_gmm_models(X0_track, labels_track, constrained=True)
fig, ax = plotting.mixture_model_grid(X0_track, labels_track, mixtures, biom_labels)
print (np.percentile(X0_track[labels_track==0][:,3],50))
#plt.show()

# select Track controls and Predict HDs for mixture models
"""
X_track = X_track[np.array(labels_track_long)==0]
lengths_track = lengths_track[labels_track==0]
labels_track = labels_track[labels_track==0]
X_predict = X[np.array(labels_long)==1]
lengths_predict = lengths[labels==1]
labels_predict = labels[labels==1]
X_track = np.vstack((X_track,X_predict))
lengths_track = np.hstack((lengths_track,lengths_predict))
labels_track = np.hstack((labels_track,labels_predict))
print (X_track.shape, lengths_track.shape, labels_track.shape)
X0_track = []
for i in range(len(lengths_track)):
    X0_track.append(X_track[np.sum(lengths_track[:i])])
X0_track = np.array(X0_track)
"""
"""
print ('Fitting '+model_type+'-'+obs_type+'-TEBM...')
if model_type == 'GMM' or model_type == 'KDE':
    if obs_type == 'Fix':
        model = tebm_fix.MixtureTEBM(X=X_predict, lengths=lengths_predict, 
                                     #                                     n_stages=n_components, time_mean=1/n_components,
                                     n_stages=n_components, time_mean=1/scale,
                                     n_iter=n_iter_inner_em, 
                                     fwd_only=False, order=n_components,
                                     algo=algorithm, verbose=False)
    else:
        model = tebm_var.MixtureTEBM(X=X_predict, lengths=lengths_predict, jumps=jumps_predict,
                                     n_components=n_components, time_mean=1, covariance_type="diag",
                                     #                                     n_iter=n_iter_inner_em, tol=tol,
                                     n_iter=1, tol=tol,
                                     transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                                     algorithm=algorithm, verbose=False, allow_nan=True,
                                     fwd_only=fwd_only, order=order)
    if obs_type == 'Fix':
        seq_model, mixtures = model.fit_tebm(labels_predict, n_start=n_start, n_iter=n_iter_seq_em, n_cores=n_cores, model_type=model_type, constrained=True, cut_controls=True)
    else:
        seq_model, mixtures = model._fit_tebm(labels_predict, n_start=n_start, n_iter=n_iter_seq_em, n_cores=n_cores, model_type=model_type, constrained=True, cut_controls=True)
        #        seq_model, mixtures = model._fit_tebm(labels_predict, n_start=n_start, n_iter=n_iter_seq_em, n_cores=n_cores, model_type=model_type, constrained=True, cut_controls=True, X_mixture=X_track, lengths_mixture=lengths_track, labels_mixture=labels_track)
elif model_type == 'Zscore':
    if obs_type == 'Fix':
        model = tebm_fix.ZscoreTEBM(X=X_predict, lengths=lengths_predict, 
                                    n_components=n_components, covariance_type="diag",
                                    n_iter=n_iter_inner_em, tol=tol,
                                    transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                                    algorithm=algorithm, verbose=False, allow_nan=False)
    else:
        model = tebm_var.ZscoreTEBM(X=X_predict, lengths=lengths_predict, jumps=jumps_predict,
                                    n_components=n_components, covariance_type="diag",
                                    n_iter=n_iter_inner_em, tol=tol,
                                    transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                                    algorithm=algorithm, verbose=False, allow_nan=False)
    seq_model = model._fit_tebm(n_zscores=n_zscores, z_max=z_max, n_start=n_start, n_iter=n_iter_seq_em, n_cores=n_cores, cut_controls=False)
else:
    print ('Likelihood model not recognised! quit()')

print (seq_model)
quit()
"""
###FIXME: HACK
#print ('HACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat, ParametricMM
from kde_ebm.distributions import Gaussian
#from kde_ebm.gmm import ParametricMM
#mixtures = fit_all_gmm_models(X0_predict, labels_predict, constrained=True)
#mixtures = fit_all_gmm_models(X0_track, labels_track, constrained=True)

pickle_file = open('./trackhd_gmms.pickle', 'rb')
load_variables = pickle.load(pickle_file)
mixtures_theta = load_variables["mixtures"]
pickle_file.close()
mixtures = []
for i,x in enumerate(mixtures_theta):
    cn_comp = Gaussian()
    ad_comp = Gaussian()
    mm = ParametricMM(cn_comp, ad_comp)
    mm.theta = x
    mixtures.append(mm)

seq_model = np.array([[1,0,2,4,3]])
###FIXME: HACK

print (seq_model)

# plots
if model_type == 'GMM' or model_type == 'KDE':
    # biomarker distributions and mixture model fits
    fig, ax = plotting.mixture_model_grid(X0_predict, labels_predict, mixtures, biom_labels)
    #    fig, ax = plotting.mixture_model_grid(X0_predict, np.ones(len(X0_predict)), mixtures, biom_labels)
    #    for i in range(X.shape[1]):
    #        print (biom_labels[i], sp.stats.spearmanr(jumps_predict, X[:,i], nan_policy='omit'))
    print (np.nanmean(X0_predict, axis=0))    
    #    plt.show()
print ('Model sequence',seq_model[0],biom_labels[seq_model[0].astype(int)])

# predictive utility
#crossval_new(dmmse, dx_change, X_predict, lengths_predict, jumps_predict, labels_predict, transmat_prior, n_components, obs_type)

# remove controls

def get_masks(lengths, labels):
    mask_short = []
    mask_long = []
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        if labels[i] == 0:
        #        if labels[i] != 2:
            mask_short.append(False)
            for j in range(nobs_i):
                mask_long.append(False)
        else:
            mask_short.append(True)
            for j in range(nobs_i):
                mask_long.append(True)
    mask_short = np.array(mask_short)
    mask_long = np.array(mask_long)
    return mask_short, mask_long

mask_short, mask_long = get_masks(lengths_predict, labels_predict)
mask_short_track, mask_long_track = get_masks(lengths_track, labels_track)
"""
X_temp, lengths_temp, jumps_temp, labels_temp, subgroup_temp, tfc_temp, dx_change_temp, times_temp, ids_temp, X0_temp, cag_temp, age_temp = [], [], [], [], [], [], [], [], [], [], [], []
for i in range(len(lengths)):
    if labels[i] != 0:
#    if labels[i] == 2:
        nobs_i = lengths[i]
        for x in X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
            X_temp.append(x)
        for x in jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
            jumps_temp.append(x)
        for x in subgroup[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
            subgroup_temp.append(x)
        for x in tfc[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
            tfc_temp.append(x)
        for x in times[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
            times_temp.append(x)            
        ids_temp.append(ids[i])
        lengths_temp.append(lengths[i])
        labels_temp.append(labels[i])
        dx_change_temp.append(dx_change[i])
        X0_temp.append(X0[i])
        cag_temp.append(cag[i])
        age_temp.append(age[i])
X = np.array(X_temp)
lengths = np.array(lengths_temp)
jumps = np.array(jumps_temp)
labels = np.array(labels_temp)
subgroup = np.array(subgroup_temp)
tfc = np.array(tfc_temp)
dx_change = np.array(dx_change_temp)
times = np.array(times_temp)
ids = np.array(ids_temp)
X0 = np.array(X0_temp)
cag = np.array(cag_temp)
age = np.array(age_temp)
print ('np.unique(labels)', np.unique(labels, return_counts=True))
"""
"""
# GPPM fit
import sys
sys.path.insert(1, '/home/pwijerat/code/GPPM/GP_progression_model_V2-develop/')
import GP_progression_model
biomarkers = ['Caudate','Putamen','motorscore','sdmt','tfc']

table = pd.read_csv('/home/paw/code/data_proc/out_data/trackhd_all_adjusted_units12months_icvcorrected_xsecreg_FORGPPM.csv')
X_gppm, Y_gppm, ids_gppm, list_biomarker, group_gppm = GP_progression_model.convert_from_df(table, biomarkers)
#X_gppm, Y_gppm, ids_gppm, list_biomarker, group_gppm, _ = GP_progression_model.convert_csv('/home/paw/code/data_proc/out_data/trackhd_all_adjusted_units12months_icvcorrected_xsecreg_FORGPPM.csv', biomarkers)

#PW hack for gppm
#Y_gppm = np.array(Y_gppm)
#for i in range(Y_gppm.shape[0]):
#    scale = np.max([y for x in Y_gppm[i].reshape(-1) for y in x])
#    if np.abs(scale) < np.abs(np.min([y for x in Y_gppm[i].reshape(-1) for y in x])):
#        scale = np.min([y for x in Y_gppm[i].reshape(-1) for y in x])
#    Y_gppm[i] /= scale

print (np.array(Y_gppm).shape, X.shape)
X_gppm = np.array(X_gppm)
X_gppm /= 4
Y_gppm = np.array(Y_gppm)
group_gppm = np.array(group_gppm)

# cut controls
X_gppm = X_gppm[:,group_gppm!=0]
Y_gppm = Y_gppm[:,group_gppm!=0]
group_gppm = group_gppm[group_gppm!=0]

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dict_monotonicity = {'Caudate': -1,
                     'Putamen': -1,
                     'motorscore': 1,
                     'sdmt': -1,
                     'tfc': -1}
model = GP_progression_model.GP_Progression_Model(X_gppm,
                                                  Y_gppm,
                                                  names_biomarkers = biomarkers,
                                                  monotonicity = [dict_monotonicity[k] for k in dict_monotonicity.keys()],
                                                  trade_off = 10,
                                                  groups = group_gppm,
                                                  group_names = ['HC','PreHD','HD'],
                                                  device = device)
model.model = model.model.to(device)

#model = GP_progression_model.GP_Progression_Model(X_gppm,
#                                                  Y_gppm,
#                                                  names_biomarkers = biom_labels,
#                                                  trade_off = np.array([-10,-10,10,-10,-10]),
#                                                  groups = group_gppm,
#                                                  group_names = ['HC','PreHD','HD'])
model.Optimize(N_outer_iterations = 6, N_iterations = 100, verbose = True, plot = False)
threshold = []
for i in range(len(model.names_biomarkers)):
    temp = []
    for x in Y_gppm[i][group_gppm==1]:
        temp.append(x[0])
    threshold.append(np.mean(temp))
    #    threshold.append(np.mean(Y[i][group==1][:,0]))
results = np.array(model.Threshold_to_time(threshold, save_fig='.', from_EBM = False))
#model.Plot(threshold=results.T[0].astype(float))
model.Plot(save_fig='.', joint = True)
"""
#transmat_prior = get_transmat_prior(n_components, order=len(biom_labels), fwd_only=True)
transmat_prior = get_transmat_prior(n_components, order=order, fwd_only=fwd_only)
print ('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Fitting TRACK model')
model = tebm_var.MixtureTEBM(X=X_predict[mask_long], lengths=lengths_predict[mask_short], jumps=jumps_predict[mask_long],
#model = tebm_var.MixtureTEBM(X=X_track[mask_long_track], lengths=lengths_track[mask_short_track], jumps=jumps_track[mask_long_track],
                             n_components=n_components, time_mean=1,
                             covariance_type="diag",
                             n_iter=n_iter_inner_em, tol=tol,
                             transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                             algorithm=algorithm, verbose=True, allow_nan=True,
                             fwd_only=fwd_only, order=order)
"""
obs_type='Fix'
model = tebm_fix.MixtureTEBM(X=X_predict, lengths=lengths_predict, 
                             n_stages=n_components, time_mean=.5,
                             n_iter=1,
                             algo=algorithm, verbose=True,
                             fwd_only=fwd_only, order=order)
"""
model.S = seq_model[0]
model.prob_mat = get_prob_mat(X_predict[mask_long], mixtures)
#model.prob_mat = get_prob_mat(X_track[mask_long_track], mixtures)
model.mixtures = mixtures
model.fit()
"""
stages_track, _ = model.predict(X=X_track[mask_long_track], lengths=lengths_track[mask_short_track], jumps=jumps_track[mask_long_track])
stages_track_bl = []
for i in range(len(lengths_track[mask_short_track])):
    stages_track_bl.append(stages_track[np.sum(lengths_track[mask_short_track][:i])])
print ([[x,y] for x,y in zip(unique_ids_track[mask_short_track],stages_track)])
print (len(unique_ids_track[mask_short_track]), len(ages_track[mask_short_track]))

stages_predict, _ = model.predict(X=X_predict[mask_long], lengths=lengths_predict[mask_short], jumps=jumps_predict[mask_long])
stages_predict_bl = []
for i in range(len(lengths_predict[mask_short])):
    stages_predict_bl.append(stages_predict[np.sum(lengths_predict[mask_short][:i])])
stages_predict_bl = np.array(stages_predict_bl)

stages_predict_bl_hdiss = []
for i in range(len(stages_predict_bl)):
    if stages_predict_bl[i]==0:
        stages_predict_bl_hdiss.append(0)
    elif stages_predict_bl[i]==1 or stages_predict_bl[i]==2:
        stages_predict_bl_hdiss.append(1)
    elif stages_predict_bl[i]==3 or stages_predict_bl[i]==4:
        stages_predict_bl_hdiss.append(2)
    elif stages_predict_bl[i]==5:
        stages_predict_bl_hdiss.append(3)
    else:
        print ('???')
stages_predict_bl_hdiss = np.array(stages_predict_bl_hdiss)

hdiss_predict = pd.read_csv('HD-ISS_stages_PREDICT-HD.csv')

hdiss_ids = [x[3:] for x in hdiss_predict['subjid']]
hdiss_predict['ID'] = hdiss_ids

hdiss_stages = []
print (len(ids), len(mask_short), len(lengths_predict))
ids = np.array(ids)
for j in range(len(ids[mask_short])):
    found = False
    for i in range(len(hdiss_predict)):
        if int(ids[mask_short][j])==int(hdiss_predict['ID'].values[i]):
            try:
                #                hdiss_stages.append(int(hdiss_predict['stage_imputed'].values[i]))
                hdiss_stages.append(int(hdiss_predict['stage_raw'].values[i]))
            except:
                hdiss_stages.append(np.nan)
            found = True
            break
    if not found:
        hdiss_stages.append(np.nan)
hdiss_stages = np.array(hdiss_stages)
print (len(hdiss_stages), len(stages_predict_bl_hdiss))
fig, ax = plt.subplots()
ax.hist([hdiss_stages[~np.isnan(hdiss_stages)], stages_predict_bl_hdiss[~np.isnan(hdiss_stages)]], stacked=True)
print (np.sum(hdiss_stages[~np.isnan(hdiss_stages)]==stages_predict_bl_hdiss[~np.isnan(hdiss_stages)])/len(hdiss_stages[~np.isnan(hdiss_stages)]))
"""
#  sojourn times
sojourns = []
sojourns.append(0)
for i in range(1,model.Q_.shape[0]-1):     # skip first stage and final stage (absorbing)
#for i in range(len(transmat)-1):     # skip first stage and final stage (absorbing)
    sojourn_i = -1/model.Q_[i,i].astype(float)/scale
    sojourns.append(sojourn_i)
    print ('Stage',i,'duration',round(sojourns[i],2),'total time so far',round(np.sum(sojourns),2))
print ('Total sequence duration',np.sum(sojourns))


# plot transmat prior
if True:
    if obs_type == 'Fix':
        transmat = model.a_mat_prior
    else:
        transmat = model.transmat_prior[0]
    fig, ax = plt.subplots()
    ax.imshow(transmat, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            if round(transmat[i, j], 3) > 1E-3:
                text = ax.text(j, i, round(transmat[i, j], 3), ha="center", va="center", color="black", size=10)
    event_labels = np.array(biom_labels)[seq_model[0].astype(int)]
    event_labels = np.insert(event_labels, 0, 'No event')
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
    #    plt.subplots_adjust(top=0.99, right=0.99, left=.21, bottom=.2)
    ax.set_title('Prior')
# initial probability
fig, ax = plt.subplots()
if obs_type == 'Fix':
    startprob = model.p_vec
else:
    startprob = model.startprob_
ax.bar(np.arange(n_components),startprob)
ax.set_xlabel('Stage', fontsize=18, labelpad=8)
ax.set_ylabel('Probability', fontsize=18, labelpad=2)
ax.tick_params(labelsize=18)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.subplots_adjust(top=0.95, right=0.99, bottom=.15)
# transition matrix
if obs_type == 'Fix':
    transmat = model.a_mat
else:
    transmat = sp.linalg.expm(np.real(scale*model.Q_))
fig, ax = plt.subplots()
ax.imshow(transmat, interpolation='nearest', cmap=plt.cm.Blues)
for i in range(transmat.shape[0]):
    for j in range(transmat.shape[1]):
        if round(transmat[i, j], 3) > 1E-3:
            text = ax.text(j, i, round(transmat[i, j], 3), ha="center", va="center", color="black", size=10)
event_labels = np.array(biom_labels)[seq_model[0].astype(int)]
event_labels = np.insert(event_labels, 0, 'No event')
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
plt.subplots_adjust(top=0.99, right=0.99, left=.21, bottom=.2)
# Q matrix
if obs_type=='Var':
    fig, ax = plt.subplots()
    ax.imshow(np.real(model.Q_), interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(model.Q_.shape[0]):
        for j in range(model.Q_.shape[1]):
            if np.abs(round(np.real(model.Q_[i, j]), 3)) > 1E-10:
                text = ax.text(j, i, round(np.real(model.Q_[i, j]), 3), ha="center", va="center", color="black", size=10)
    event_labels = np.array(biom_labels)[seq_model[0].astype(int)]
    event_labels = np.insert(event_labels, 0, 'No event')
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
    plt.subplots_adjust(top=0.99, right=0.99, left=.21, bottom=.2)
#plt.show()
# convert stages to time-to-onset, defined as reaching MMSE abnormality
"""
dx_change_time_model = []
for i in range(len(stages_model_bl)):
    dx_change_time_model.append(sojourns[8]-sojourns[stages_model_bl[i]]+np.random.rand())
    #    dx_change_time_model.append(sojourns[stages_model_bl[i]])
dx_change_time_model = np.array(dx_change_time_model)
print (dx_change_time.shape,dx_change_time_model.shape)
dx_change_time_model = dx_change_time_model[labels==2]
dx_change_time_model = dx_change_time_model[dx_change[labels==2]==1]
dx_change_time = dx_change_time[labels==2]
dx_change_time = dx_change_time[dx_change[labels==2]==1]
dx_change_time /= 4
print (dx_change_time,dx_change_time_model)
print (dx_change_time.shape,dx_change_time_model.shape)
fig, ax = plt.subplots()
ax.scatter(dx_change_time, dx_change_time_model)
print (sp.stats.pearsonr(dx_change_time[~np.isinf(dx_change_time)], dx_change_time_model[~np.isinf(dx_change_time)]))
print (sp.stats.spearmanr(dx_change_time[~np.isinf(dx_change_time)], dx_change_time_model[~np.isinf(dx_change_time)]))
"""

# bootstrap uncertainty
time_prior_a, time_prior_b = .5, 1.5
"""
transmat_boot_arr, Q_boot_arr, seq_boot_arr = [], [], []
for b in range(100):
    print ('boot',b+1)
    # generate bootstrap sample
    X_boot, jumps_boot, labels_boot, lengths_boot, k_boot = [], [], [], [], []
    if False:
        # simulated bootstrap
        for i in range(len(lengths)):
            X_i, k_i, dt_i, label_i = model.gen_sample(n_samples=lengths[i], scale=scale)
            lengths_boot.append(lengths[i])
            for j in range(len(X_i)):
                X_boot.append(X_i[j])
                k_boot.append(k_i[j])
                jumps_boot.append(dt_i[j])
            labels_boot.append(label_i)
        X_boot = np.array(X_boot)
        jumps_boot = np.array(jumps_boot)
        labels_boot = np.array(labels_boot)
        lengths_boot = np.array(lengths_boot)
        k_boot = np.array(k_boot)
    else:
        # data bootstrap
        #        ids_boot = resample(ids, stratify=labels)
        
        for i in range(len(lengths)):
            idx = np.random.randint(len(lengths))
            #            idx = np.where(np.array(ids)==str(ids_boot[i]))[0][0]
            nobs_idx = lengths[idx]
            lengths_boot.append(nobs_idx)
            labels_boot.append(labels[idx])
            s_idx, e_idx = int(np.sum(lengths[:idx])), int(np.sum(lengths[:idx])+nobs_idx)
            X_idx = X[s_idx:e_idx]
            jumps_idx = jumps[s_idx:e_idx]
            for j in range(len(X_idx)):
                X_boot.append(X_idx[j])
                jumps_boot.append(jumps_idx[j])
        X_boot = np.array(X_boot)
        lengths_boot = np.array(lengths_boot)
        labels_boot = np.array(labels_boot)
        jumps_boot = np.array(jumps_boot)

        X_boot_track, labels_boot_track, lengths_boot_track = [], [], []
        for i in range(len(lengths_track)):
            idx = np.random.randint(len(lengths_track))
            nobs_idx = lengths_track[idx]
            lengths_boot_track.append(nobs_idx)
            labels_boot_track.append(labels_track[idx])
            s_idx, e_idx = int(np.sum(lengths_track[:idx])), int(np.sum(lengths_track[:idx])+nobs_idx)
            X_idx = X_track[s_idx:e_idx]
            for j in range(len(X_idx)):
                X_boot_track.append(X_idx[j])
        X_boot_track = np.array(X_boot_track)
        lengths_boot_track = np.array(lengths_boot_track)
        labels_boot_track = np.array(labels_boot_track)

        #        X0_boot = []
        #        for i in range(len(lengths)):
        #            X0_boot.append(X_boot[np.sum(lengths[:i])])
        #        X0_boot = np.array(X0_boot)
        #        mixtures_boot = fit_all_gmm_models(X0_boot, labels_boot)

    if False:#plot_all_data:
        n_x = np.round(np.sqrt(n_bms)).astype(int)
        n_y = np.ceil(np.sqrt(n_bms)).astype(int)
        fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
        for i in range(n_bms):
            for j in range(len(lengths)):
                nobs_i = lengths[j]
                s_idx, e_idx = int(np.sum(lengths[:j])), int(np.sum(lengths[:j])+nobs_i)
                #if i==0:
                #    print (jumps_boot[s_idx:e_idx],np.cumsum(jumps_boot[s_idx:e_idx]),X_boot[s_idx:e_idx,i])
                ax[i // n_x, i % n_x].plot(np.cumsum(jumps_boot[s_idx:e_idx]),X_boot[s_idx:e_idx,i])
                ax[i // n_x, i % n_x].set_title(biom_labels[i])
    if False:
        # biomarker distributions and mixture model fits
        #        fig, ax = plotting.mixture_model_grid(X0_boot, np.ones(len(X0_boot)), mixtures, biom_labels)
        #        fig, ax = plotting.mixture_model_grid(X0_boot, labels_boot, mixtures_boot, biom_labels)
        for i in range(X_boot.shape[1]):
            print (biom_labels[i], sp.stats.spearmanr(jumps_boot, X_boot[:,i], nan_policy='omit'))
        print (np.mean(X0_boot, axis=0))

    # uniform prior on prior; minimum transition time should be q_ii=1, maximum is up to user
    # could also investigate Dirichlet prior on Q
    time_mean_boot = time_prior_a + np.random.rand()*time_prior_b

    model_boot = tebm_var.MixtureTEBM(
        X=X_boot,
        lengths=lengths_boot,
        jumps=jumps_boot,
        n_components=n_components,
        time_mean=1/scale,
        covariance_type="diag",
        n_iter=1,
        tol=tol,
        transmat_prior=transmat_prior,
        init_params=init_params,
        params=fit_params,
        algorithm=algorithm,
        verbose=False,
        allow_nan=True,
        fwd_only=fwd_only,
        order=order)
    seq_model_boot, mixtures_boot = model_boot._fit_tebm(labels_boot, n_start=4, n_iter=n_iter_seq_em, n_cores=n_cores, model_type=model_type, constrained=False, cut_controls=False, X_mixture=X_boot_track, lengths_mixture=lengths_boot_track, labels_mixture=labels_boot_track)

    mask_short_boot, mask_long_boot = get_masks(lengths_boot, labels_boot)
    # refit model
    model_boot = tebm_var.MixtureTEBM(
        X=X_boot[mask_long_boot],
        lengths=lengths_boot[mask_short_boot],
        jumps=jumps_boot[mask_long_boot],
        n_components=n_components,
        time_mean=1/scale,
        covariance_type="diag",
        n_iter=n_iter_inner_em,
        tol=tol,
        transmat_prior=transmat_prior,
        init_params=init_params,
        params=fit_params,
        algorithm=algorithm,
        verbose=True,
        allow_nan=True,
        fwd_only=fwd_only,
        order=order)
    # use bootstrapped model parameters
    model_boot.S = seq_model_boot[0]
    model_boot.prob_mat = get_prob_mat(X_boot[mask_long_boot], mixtures_boot)
    model_boot.mixtures = mixtures_boot    
    # use ML model parameters
#    model_boot.S = seq_model[0]
#    model_boot.prob_mat = get_prob_mat(X_boot[mask_long_boot], mixtures)
#    model_boot.mixtures = mixtures    
    model_boot.fit()
    if obs_type == 'Fix':
        transmat_boot = model_boot.a_mat
    else:
        transmat_boot = sp.linalg.expm(scale*np.real(model_boot.Q_))
        Q_boot_arr.append(np.real(model_boot.Q_))
    transmat_boot_arr.append(transmat_boot)
    seq_boot_arr.append(seq_model_boot[0])

Q_boot_arr = np.array(Q_boot_arr)
transmat_boot_arr = np.array(transmat_boot_arr)
seq_boot_arr = np.array(seq_boot_arr)

save_variables = {}
save_variables["X"] = X_predict
save_variables["lengths"] = lengths_predict
save_variables["jumps"] = jumps_predict
save_variables["labels_predict"] = labels_predict
save_variables["seq_model"] = seq_model
save_variables["startprob"] = model.startprob_
save_variables["transmat"] = transmat
save_variables["Q"] = model.Q_
save_variables["transmat_boot_arr"] = transmat_boot_arr
save_variables["Q_boot_arr"] = Q_boot_arr
save_variables["seq_boot_arr"] = seq_boot_arr
#pickle_file = open('./test_5simboots_Sboot_Nbms'+str(X.shape[1])+'_Nits'+str(n_iter_inner_em)+'_prior1_'+data_out, 'wb')
pickle_file = open('./predict_test_100boots_Nbms'+str(X.shape[1])+'_Nits'+str(n_iter_inner_em)+'_prior1_'+data_out, 'wb') 
pickle_output = pickle.dump(save_variables, pickle_file)
pickle_file.close()
"""

pickle_file = open('./predict_test_100boots_Nbms'+str(X_predict.shape[1])+'_Nits'+str(n_iter_inner_em)+'_prior1_'+data_out, 'rb')
load_variables = pickle.load(pickle_file)
transmat_boot_arr = np.array(load_variables["transmat_boot_arr"])
Q_boot_arr = np.array(load_variables["Q_boot_arr"])
seq_boot_arr = np.array(load_variables["seq_boot_arr"])
pickle_file.close()

"""
def get_seq_freq(seq):
    temp = []
    for i in range(len(seq)):
        count = 1
        seen = False
        if len(temp) > 0:
            for x in np.array(temp).T[0]:
                if (x == seq[i]).all():
                    seen = True
        if seen:
            continue
        for j in range(i+1,len(seq)):
            if (seq[i] == seq[j]).all():
                count += 1
        temp.append([seq[i], count])
    return np.array(temp)
seq_freq = get_seq_freq(seq_boot_arr)
print (seq_model, seq_freq[np.argmax(seq_freq.T[1])].T[0], seq_freq)
"""
fig, ax = plt.subplots()
temp = []
del_idx = []
Q_boot_arr_temp = []
for i in range(len(Q_boot_arr)):
#    if (Q_boot_arr[i].diagonal()<-10).any():
#    if np.sum([-1/x for x in Q_boot_arr[i].diagonal()[1:-1]]) < 10:
    if False:
        print ('deleting!',np.sum([-1/x for x in Q_boot_arr[i].diagonal()[1:-1]]))
    else:
        #        temp.append(np.sum([-1/x for x in Q_boot_arr[i].diagonal()[1:-1]]))
#        temp.append(-1/Q_boot_arr[i].diagonal()[1])
        temp.append(Q_boot_arr[i].diagonal()[1])
#        temp.append(np.log(-Q_boot_arr[i].diagonal()[1]))
        Q_boot_arr_temp.append(Q_boot_arr[i])
print (np.median(temp), np.std(temp), np.min(temp), np.max(temp))
ax.hist(temp, bins=1000)
ax.set_xlabel('Bootstrapped total durations')

df_out = pd.DataFrame(data=temp,columns=['Q_1'])
df_out.to_csv('Q_out.csv',index=False)

###
print (Q_boot_arr.shape)
Q_boot_arr = np.array(Q_boot_arr_temp)
print (Q_boot_arr.shape)
#Q_boot_arr = np.sort(Q_boot_arr[:,1,1])
#print (1/Q_boot_arr[25], 1/Q_boot_arr[975])

#print (Q_boot_arr[0], model.Q_)
#quit()
###

transmat_mean = np.mean(transmat_boot_arr, axis=0)
transmat_std = np.std(transmat_boot_arr, axis=0)
Q_boot_mean = np.mean(Q_boot_arr, axis=0)
Q_boot_std = np.std(Q_boot_arr, axis=0)

#  sojourn times
sojourns, sojourns_err = [], []
sojourns.append(0)
for i in range(1,len(transmat)-1):     # skip first stage and final stage (absorbing)
#for i in range(len(transmat)-1):     # skip first stage and final stage (absorbing)
    if obs_type=='Var':
        sojourn_i = -1/model.Q_[i,i].astype(float)/scale
        #        sojourn_i = -1/Q_boot_mean[i,i].astype(float)/scale
        # relative error
        sojourn_i_err = -Q_boot_std[i,i]/Q_boot_mean[i,i]
    else:
#        sojourn_i = 1/(1-model.a_mat[i,i])
        sojourn_i = 1/(1-transmat_mean[i,i])
        # relative error
        sojourn_i_err = transmat_std[i,i]/model.a_mat[i,i]
    sojourns.append(sojourn_i)
    sojourns_err.append(sojourn_i_err)
    print ('Stage',i,'duration',round(sojourns[i],2),'+/-',round(sojourns[i]*sojourn_i_err,2),
           'total time so far',round(np.sum(sojourns),2))
print ('Total sequence duration',np.sum(sojourns),'+/-',np.sum(sojourns)*np.sqrt(np.mean(np.power(sojourns_err,2))))

# calculate CIs
confints_x, confints_x_cum = [], []
confints_x.append([0,0])
confints_x_cum.append([0,0])

# empirical bootstrap
# (see https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/77906546c17ee79eb6e64194175e82ed_MIT18_05S14_Reading24.pdf)
for i in range(1,Q_boot_arr.shape[1]-1):
    # data to resample
    Q_vec = Q_boot_arr[:,i,i].astype(float)

#    Q_vec = Q_vec[Q_vec<-1E-2]    
#    Q_vec = np.array([-1/x for x in Q_vec])
    
    print (np.mean(Q_vec))
    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='bca')
    print (ci.confidence_interval)
#    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='percentile')
#    print (ci.confidence_interval)
#    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='basic')
#    print (ci.confidence_interval)

    #    Q_vec = Q_vec[Q_vec<-0.05]
    #    Q_vec = Q_vec[Q_vec>-4]
    
    # bootstrap with replacement samples of same length
#    Q_vec_boot = np.array([np.mean(resample(Q_vec)) for i in range(10000)])
    Q_vec_boot = Q_vec
    # subtract data mean
    Q_mean = np.mean(Q_vec)
#    Q_mean = model.Q_[i,i].astype(float) # could be ML fit mean? don't think so, since it should be the mean of the sample used for resampling

#    fig, ax = plt.subplots()
#    ax.hist(Q_vec_boot)
    
#    Q_vec_boot -= Q_mean

#    fig, ax = plt.subplots()
#    ax.hist(Q_vec_boot)
#    plt.show()
    
    # calculate percentiles
    #    d_i = np.abs(np.array([np.percentile(Q_vec_boot, 2.5), np.percentile(Q_vec_boot, 97.5)])) # [-,+]
    #    d_i = np.array([1.96*np.std(Q_vec)/np.sqrt(len(Q_vec)), 1.96*np.std(Q_vec)/np.sqrt(len(Q_vec))])
    d_i = np.array([ci.confidence_interval[0], ci.confidence_interval[1]])

    
    print ('!',Q_mean, d_i)
#    fig, ax = plt.subplots()
#    ax.hist(Q_vec)
#    ax.axvline(model.Q_[i,i].astype(float))
#    plt.show()
    # convert to unit time
    #    x_upper = -1/(Q_mean-d_i[0])
    #    x_lower = -1/(Q_mean-d_i[1])

    #    x_mean = -1/model.Q_[i,i].astype(float)
    #    x_upper = x_mean*d_i[0]/Q_mean
    #    x_lower = x_mean*d_i[1]/Q_mean
    """
    mean_log = np.log(-Q_mean)
    stderr_log = ci.standard_error*1/(-Q_mean)
    x_mean = np.exp(mean_log)
    x_upper = np.exp(mean_log + 1.96*stderr_log)
    x_lower = np.exp(mean_log - 1.96*stderr_log)
    print (x_lower, x_mean, x_upper)
    """
    
    #    confints_x.append([x_mean-x_lower, x_upper-x_mean]) # [-,+]
    #    confints_x.append([Q_mean-d_i[0], d_i[1]-Q_mean]) # [-,+]
    confints_x.append([-1/model.Q_[i,i].astype(float)*d_i[1]/Q_mean, -1/model.Q_[i,i].astype(float)*d_i[0]/Q_mean]) # [-,+]
    print (confints_x[i])
    # cumulative uncertainty
    confints_x_cum.append(np.sqrt(np.sum(np.power(confints_x,2),axis=0)))
confints_x = np.array(confints_x)
# cumulative error = RMS of all previous errors
confints_x_cum = np.array(confints_x_cum)
# cumulative error = sum of all previous errors
#confints_x_cum = np.cumsum(confints_x, axis=0)
# dummy errors on y; could make this equal to event sequence uncertainty
confints_y = np.ones(confints_x.shape)*.5
confints_y[0] = 0

"""
sojourns = np.array(sojourns)
sojourns /= 2
confints_x /= 2
confints_x_cum /= 2
"""
#
fig, ax = plt.subplots()
sojourns_boot = []
for i in range(Q_boot_arr.shape[1]-1):
    sojourns_boot.append([-1/x/scale for x in Q_boot_arr[:,i,i]])
    if i>0:
        sojourns_boot[i] = [x+np.mean(sojourns_boot[i-1]) for x in sojourns_boot[i]]
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
              for x, y, xe, ye in zip(np.cumsum(sojourns), np.arange(len(sojourns)), confints_x_cum, confints_y)]
colours = [colors[0],colors[0],colors[1],colors[1],colors[1]]
#colours = [colors[0],colors[0],colors[1],colors[2],colors[3]]
pc1 = PatchCollection(errorboxes, facecolor=colours, alpha=0.5, edgecolor='black', linestyle='--', hatch='//')
ax.add_collection(pc1)
ax.scatter(np.cumsum(sojourns), np.arange(len(sojourns)), c='black', s=20)
eb1 = ax.errorbar(np.cumsum(sojourns), np.arange(len(sojourns)), xerr=confints_x.T, capsize=5, capthick=2, ls='none', c='black')
eb1[-1][0].set_linestyle('--')
ax.set_yticks(np.arange(len(sojourns)))
ax.set_yticklabels(np.array(biom_labels)[seq_model[0].astype(int)], ha='right', rotation_mode='anchor')
#ax.set_yticklabels(event_labels, ha='right', rotation_mode='anchor')
ax.set_xlabel('Disease time (years)', fontsize=16)
#ax.set_xlim(-.5,36)
#ax.set_xlim(-.5,20)
ax.set_ylim(0,len(biom_labels))
ax.tick_params(labelsize=16)
ax.grid(axis='y')

# cumulative time
sojourns = np.cumsum(sojourns)
"""
confints_x_cum_copy = confints_x_cum.copy()
confints_x_cum_copy[:,0] *= -1
for x in [x+y for x,y in zip(confints_x_cum_copy,sojourns)]:
    print ([round(y,1) for y in x])
for i in range(1,len(sojourns)):
    tval = str(round(sojourns[i],1))
    if tval[-1]=='0':
        tval = tval[:-2]
    ax.text(sojourns[i],len(biom_labels)+.55,tval,fontsize=16, bbox=dict(facecolor='none', edgecolor='black', linestyle='--'))
    plt.axvline(sojourns[i], ymin=i/len(sojourns), ymax=1.09, color='black', alpha=0.5, linewidth=2, linestyle='--', clip_on=False)
"""
#ax.text(9.2,len(biom_labels)+2,'Event time (years)',fontsize=16)

# add Track
pickle_file = open('./test_100boots_Nbms'+str(X_track.shape[1])+'_Nits'+str(n_iter_inner_em)+'_prior1_tebm_trackhd_adjusted.pickle', 'rb')
load_variables = pickle.load(pickle_file)
Q_boot_arr_track = np.array(load_variables["Q_boot_arr"])
pickle_file.close()

model_track = tebm_var.MixtureTEBM(X=X_track[mask_long_track], lengths=lengths_track[mask_short_track], jumps=jumps_track[mask_long_track],
                             n_components=n_components, time_mean=1,
                             covariance_type="diag",
                             n_iter=n_iter_inner_em, tol=tol,
                             transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                             algorithm=algorithm, verbose=True, allow_nan=True,
                             fwd_only=fwd_only, order=order)
model_track.S = seq_model[0]
model_track.prob_mat = get_prob_mat(X_track[mask_long_track], mixtures)
model_track.mixtures = mixtures
model_track.fit()

Q_track = model_track.Q_#np.array(load_variables["Q"])

sojourns_track = []
sojourns_track.append(0)
for i in range(1,Q_track.shape[0]-1):     # skip first stage and final stage (absorbing)
    sojourn_i = -1/Q_track[i,i].astype(float)/scale
    sojourns_track.append(sojourn_i)
confints_x, confints_x_cum = [], []
confints_x.append([0,0])
confints_x_cum.append([0,0])
for i in range(1,Q_boot_arr_track.shape[1]-1):
    Q_vec = Q_boot_arr_track[:,i,i].astype(float)
    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='bca')
    Q_vec_boot = Q_vec
    # subtract data mean
    Q_mean = np.mean(Q_vec)
    d_i = np.array([ci.confidence_interval[0], ci.confidence_interval[1]])
    confints_x.append([-1/model_track.Q_[i,i].astype(float)*d_i[1]/Q_mean, -1/model_track.Q_[i,i].astype(float)*d_i[0]/Q_mean]) # [-,+]
    # cumulative uncertainty
    confints_x_cum.append(np.sqrt(np.sum(np.power(confints_x,2),axis=0)))
confints_x = np.array(confints_x)
confints_x_cum = np.array(confints_x_cum)
confints_y = np.ones(confints_x.shape)*.5
confints_y[0] = 0
errorboxes_track = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
                    for x, y, xe, ye in zip(np.cumsum(sojourns_track), np.arange(len(sojourns_track)), confints_x_cum, confints_y)]
pc2 = PatchCollection(errorboxes_track, facecolor=colours, alpha=0.5, edgecolor='black')
ax.add_collection(pc2)
ax.scatter(np.cumsum(sojourns_track), np.arange(len(sojourns_track)), c='black', s=20)
ax.errorbar(np.cumsum(sojourns_track), np.arange(len(sojourns_track)), xerr=confints_x.T, capsize=5, capthick=2, ls='none', c='black')
# cumulative time
sojourns_track = np.cumsum(sojourns_track)
confints_x_cum_copy = confints_x_cum.copy()
confints_x_cum_copy[:,0] *= -1
for x in [x+y for x,y in zip(confints_x_cum_copy,sojourns_track)]:
    print ([round(y,1) for y in x])
for i in range(1,len(sojourns_track)):
    tval = str(round(sojourns_track[i],1))
    if tval[-1]=='0':
        tval = tval[:-2]
    if i < (len(sojourns_track)-1):
        ax.text(sojourns_track[i]-.75,len(biom_labels)+.2,tval,fontsize=16, bbox=dict(facecolor='none', edgecolor='black'))
        plt.axvline(sojourns_track[i], i/len(sojourns_track), color='black', alpha=0.5, linewidth=2)
    else:
        ax.text(sojourns_track[i]+.5,len(biom_labels)+.2,tval,fontsize=16, bbox=dict(facecolor='none', edgecolor='black'))
        plt.axvline(sojourns_track[i], i/len(sojourns_track), color='black', alpha=0.5, linewidth=2)
#plt.show()
##################################CUT CONTROLS
print ('Cutting controls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
## remake masks, now only select PreHDs
def get_masks(lengths, labels):
    mask_short = []
    mask_long = []
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        #        if labels[i] == 0:
        if labels[i] != 2:
            mask_short.append(False)
            for j in range(nobs_i):
                mask_long.append(False)
        else:
            mask_short.append(True)
            for j in range(nobs_i):
                mask_long.append(True)
    mask_short = np.array(mask_short)
    mask_long = np.array(mask_long)
    return mask_short, mask_long
mask_short, mask_long = get_masks(lengths_predict, labels_predict)

X_predict = X_predict[mask_long]
lengths_predict = lengths_predict[mask_short]
jumps = jumps_predict[mask_long]
labels_predict = labels_predict[mask_short]
X0_predict = X0_predict[mask_short]
times_predict = times_predict[mask_long]
ids = np.array(ids)[mask_short]
cag = cag[mask_short]
age = age[mask_short]


cap = np.array([(x-33.66)*y for x,y in zip(cag,age)])

delta_t = 2
stages_model, _ = model_track.predict(X_predict, lengths_predict, jumps_predict)
#FIXME: should be using all data to stage here?
stages_model_bl = []
for i in range(len(lengths_predict)):
    stages_model_bl.append(stages_model[int(np.sum(lengths_predict[:i]))])
stages_model_bl = np.array(stages_model_bl)
pred_0, pred_1 = tebm_preds(model_track, X_predict, lengths_predict, jumps_predict, stages_model, 1, obs_type, scale*delta_t)


idx = np.where(labels_predict==2)[0][1]
from utils import pred_traj
print (len(sojourns))
master_arr = pred_traj(model, X0_predict, [stages_model_bl[idx]], lengths_predict, jumps_predict, idx, sojourns)

stage_mode_pred = np.array([x[0]-1 for x in master_arr]) # -1 because we omit stage 0 from the plot
time_mean_pred = np.array([x[1] for x in master_arr])
time_std_pred = np.array([x[2] for x in master_arr])
stage_mode_pred = stage_mode_pred[~np.isnan(time_mean_pred)]
time_std_pred = time_std_pred[~np.isnan(time_mean_pred)]
time_mean_pred = time_mean_pred[~np.isnan(time_mean_pred)]
print (stage_mode_pred, time_std_pred, time_mean_pred)

#line, = ax.plot(time_mean_pred, stage_mode_pred, color='red', linestyle='--', alpha=.5, label='Patient X')
#eb1 = ax.errorbar(time_mean_pred, stage_mode_pred, xerr=time_std_pred, capsize=5, capthick=2, c='red', ls='--', alpha=.5)
#eb1[-1][0].set_linestyle('--')


from matplotlib.patches import Patch
#legend_elements = [line,
#                   Patch(facecolor='none', edgecolor='black', label='TRACK-HD'),
#                   Patch(facecolor='none', hatch='///', edgecolor='black', linestyle='--', label='PREDICT-HD')]
legend_elements = [Patch(facecolor='none', edgecolor='black', label='TRACK-HD'),
                   Patch(facecolor='none', hatch='///', edgecolor='black', linestyle='--', label='PREDICT-HD')]
ax.legend(handles=legend_elements, prop={'size': 16}, loc='lower right')


###
fig, ax = plt.subplots()
pc = PatchCollection(errorboxes_track, facecolor=colours, alpha=0.5, edgecolor='black')
ax.add_collection(pc)

stages_model_predict, _ = model_track.predict(X_predict, lengths_predict, jumps_predict)
stages_model_predict_bl, _ = model_track.predict(X0_predict, np.array([1]*len(X0_predict)), np.array([0]*len(X0_predict)))

###PREDICT-HD predicted trajectories
residual_predict = []
print (len(times_predict), len(stages_model_predict_bl), len(lengths_predict))
for i in range(len(stages_model_predict_bl)):
    
    if labels_predict[i]==0 or stages_model_predict_bl[i]==0 or stages_model_predict_bl[i]==len(sojourns):
        continue
    
    master_arr = pred_traj(model_track, X0_predict, [stages_model_predict_bl[i]], lengths_predict, jumps_predict, i, sojourns)

    stage_pred = np.array([x[0]-1 for x in master_arr]) # -1 because we omit stage 0 from the plot
    time_mean_pred = np.array([x[1] for x in master_arr])
    time_std_pred = np.array([x[2] for x in master_arr])
    stage_pred = stage_pred[~np.isnan(time_mean_pred)]
    time_std_pred = time_std_pred[~np.isnan(time_mean_pred)]
    time_mean_pred = time_mean_pred[~np.isnan(time_mean_pred)]
    #    print (stage_pred, time_std_pred, time_mean_pred)
    
    line, = ax.plot(time_mean_pred, stage_pred, color='grey', linestyle='--', linewidth=2, alpha=.5, label='PREDICT-HD PreHD (predicted)')
    eb1 = ax.errorbar(time_mean_pred, stage_pred, xerr=time_std_pred, capsize=5, capthick=2, c='grey', ls='--', alpha=.5)
    eb1[-1][0].set_linestyle('--')
    ax.scatter(time_mean_pred[0], stage_pred[0], c='grey', s=20)
    eb2 = ax.errorbar(time_mean_pred[0], stage_pred[0], xerr=time_std_pred[0], capsize=5, capthick=2, c='grey', ls='--')
    eb2[-1][0].set_linestyle('--')

    stage_pred += 1 # re-add one for comparison with staging
    nobs_i = lengths_predict[i]
    s_idx, e_idx = int(np.sum(lengths_predict[:i])), int(np.sum(lengths_predict[:i])+nobs_i)
    times_i = times_predict[s_idx:e_idx]
    stages_model_i = stages_model_predict[s_idx:e_idx]
    for j in range(len(stages_model_i)):
        if stages_model_i[j] > stages_model_i[0] and stages_model_i[j] in stage_pred:
            idx = np.where(stages_model_i[j]==stage_pred)[0][0]
            time_obs = times_i[j]
            time_pred = time_mean_pred[idx]-sojourns[int(stage_pred[0])]
            #            print (time_obs, time_pred)
            #            print (stages_model_i[j], stage_pred[0])
            residual_predict.append(time_obs-time_pred)
    """
    for j in range(1,len(time_mean_pred)): # skip first event because it's not predicted
        #        residual_predict.append(sojourns[int(stage_pred[j])]-time_mean_pred[j])
        if stage_pred[j] in list(stages_model_i):
            print (stage_pred[j], stage_pred, stages_model_i)
            idx = np.where(stage_pred[j]==stages_model_i)[0][0]
            #            time_obs = np.cumsum(times_i)[idx]
            time_obs = times_i[idx]
            print (idx, int(stage_pred[0]))
            print (time_obs, time_mean_pred[j]-sojourns[int(stage_pred[0])])
            residual_predict.append(time_obs-time_mean_pred[j]-sojourns[int(stage_pred[0])])
    """
print (np.mean(residual_predict), np.median(residual_predict), np.std(residual_predict))
ax.set_yticks(np.arange(len(sojourns_track)))
ax.set_yticklabels(np.array(biom_labels)[seq_model[0].astype(int)], ha='right', rotation_mode='anchor')
ax.set_xlabel('Disease time (years)', fontsize=16)
#ax.set_xlim(-.5,27.3)
ax.set_ylim(0,len(biom_labels))
ax.tick_params(labelsize=16)
ax.grid(axis='y')
legend_elements = [Patch(facecolor='none', edgecolor='black', label='TRACK-HD')]
legend_elements.append(line)
ax.legend(handles=legend_elements, prop={'size': 16}, loc='lower right')

fig, ax = plt.subplots()
my_dict = {'PREDICT-HD':residual_predict}
ax.boxplot(my_dict.values(), showmeans=True)
ax.tick_params(length=10,labelsize='large')
ax.set_xticklabels(my_dict.keys(), fontsize=20)
ax.set_ylabel('Time-to-event residual (years)', fontsize=20)

"""
###PREDICT predicted trajectories
#idx = np.where(labels_predict==2)[0][0]
for i in range(len(stages_model_predict_bl)):
    print (stages_model_predict_bl[i])
    if labels_predict[i]==0 or stages_model_predict_bl[i]==0 or stages_model_predict_bl[i]==len(sojourns):
        continue
    
    master_arr = pred_traj(model_track, X0_predict, [stages_model_predict_bl[i]], lengths_predict, jumps_predict, i, sojourns)

    stage_mode_pred = np.array([x[0]-1 for x in master_arr]) # -1 because we omit stage 0 from the plot
    time_mean_pred = np.array([x[1] for x in master_arr])
    time_std_pred = np.array([x[2] for x in master_arr])
    stage_mode_pred = stage_mode_pred[~np.isnan(time_mean_pred)]
    time_std_pred = time_std_pred[~np.isnan(time_mean_pred)]
    time_mean_pred = time_mean_pred[~np.isnan(time_mean_pred)]
    print (stage_mode_pred, time_std_pred, time_mean_pred)

    line, = ax.plot(time_mean_pred, stage_mode_pred, color='red', linestyle='--', linewidth=2, alpha=.5, label='PREDICT-HD PreHD (prediction)')
    eb1 = ax.errorbar(time_mean_pred, stage_mode_pred, xerr=time_std_pred, capsize=5, capthick=2, c='red', ls='--', alpha=.5)
    eb1[-1][0].set_linestyle('--')
#from matplotlib.patches import Patch
legend_elements.append(line)
ax.legend(handles=legend_elements, prop={'size': 16}, loc='lower right')
"""
"""
from utils import pred_traj
print (len(stages_model_bl), len(pred_0))
fig, ax = plt.subplots()
colors = []
cag_unique = np.unique(cag)
for i in range(len(cag_unique)):
    colors.append([1.0,i*200./len(cag_unique)/255.,0])
colors = colors[::-1]
lines, seen = [], []
rate_cag, rate_cap = [], []
for i in range(len(stages_model_bl)):
#for i in range(10):

    #    if cag[i] in seen:
    #       continue 
    
    if stages_model_bl[i]==0 or stages_model_bl[i]==len(sojourns):        
        continue
    
    master_arr = pred_traj(model, X0, [stages_model_bl[i]], lengths, jumps, i, sojourns)
    print (master_arr)
    stage_mode_pred = np.array([x[0]-1 for x in master_arr]) # -1 because we omit stage 0 from the plot
    time_mean_pred = np.array([x[1] for x in master_arr])
    time_std_pred = np.array([x[2] for x in master_arr])
    stage_mode_pred = stage_mode_pred[~np.isnan(time_mean_pred)]
    time_std_pred = time_std_pred[~np.isnan(time_mean_pred)]
    time_mean_pred = time_mean_pred[~np.isnan(time_mean_pred)]
    idx_clr = np.where(cag[i]==cag_unique)[0][0]
    plt_clr = (colors[idx_clr][0],colors[idx_clr][1],colors[idx_clr][2])
    line, = ax.plot(time_mean_pred, stage_mode_pred, color=plt_clr, linestyle='--', label='CAG = '+str(int(cag[i])))
    if not cag[i] in seen:
        lines.append(line)
        seen.append(cag[i])
    eb1 = ax.errorbar(time_mean_pred, stage_mode_pred, xerr=time_std_pred, capsize=5, capthick=2, c=plt_clr, ls='--')
    eb1[-1][0].set_linestyle('--')    
    rate_cag.append([(stage_mode_pred[-1]-stage_mode_pred[0])/(time_mean_pred[-1]-time_mean_pred[0]), cag[i]])
    rate_cap.append([(stage_mode_pred[-1]-stage_mode_pred[0])/(time_mean_pred[-1]-time_mean_pred[0]), cap[i]])
ax.legend(handles=lines, prop={'size': 16})

fig, ax = plt.subplots()
rate_cag = np.array(rate_cag)
ax.scatter(rate_cag[:,1], rate_cag[:,0])
print (sp.stats.spearmanr(rate_cag[:,1], rate_cag[:,0], nan_policy='omit'))

fig, ax = plt.subplots()
rate_cap = np.array(rate_cap)
ax.scatter(rate_cap[:,1], rate_cap[:,0])
print (sp.stats.spearmanr(rate_cap[:,1], rate_cap[:,0], nan_policy='omit'))
"""
"""
colors = []
for i in range(len(np.unique(stages_model_bl))):
    colors.append([1.0,i*200./len(np.unique(stages_model_bl))/255.,0])
colors = colors[::-1]
fig, ax = plt.subplots()
sctr_clrs = []

pred_0 = pred_0[stages_model_bl<(len(sojourns)-1)]
cap = cap[stages_model_bl<(len(sojourns)-1)]
stages_model_bl = stages_model_bl[stages_model_bl<(len(sojourns)-1)]

pred_0 = pred_0[stages_model_bl!=0]
cap = cap[stages_model_bl!=0]
stages_model_bl = stages_model_bl[stages_model_bl!=0]


for i in range(len(stages_model_bl)):
    sctr_clrs.append(colors[int(stages_model_bl[i])])
ax.scatter(pred_0, cap, c=sctr_clrs)
print (sp.stats.spearmanr(pred_0, cap, nan_policy='omit'))
"""
#plt.show()

pred_0 = pred_0[~np.isnan(cag)]
stages_model_bl = stages_model_bl[~np.isnan(cag)]
cag = cag[~np.isnan(cag)]


####CUT
#pred_0 = pred_0[stages_model_bl==2]
#cag = cag[stages_model_bl==2]
#stages_model_bl = stages_model_bl[stages_model_bl==2]

cap = np.array([(x-33.66)*y for x,y in zip(cag,age)])
fig, ax = plt.subplots()
ax.hist(cap)
#cag_cut = 42
cap_cut = 400
print (pred_0[cap<=cap_cut].shape, pred_0[cap>cap_cut].shape)

fig, ax = plt.subplots()
ax.scatter(pred_0[pred_0>0.1], cap[pred_0>0.1])
print (sp.stats.spearmanr(pred_0[pred_0>0.1], cap[pred_0>0.1]))
#plt.show()

fig, ax = plt.subplots()
#ax.hist([pred_0[cap<=cap_cut],pred_0[cap>cap_cut]], stacked=True, label=['CAP <= '+str(cap_cut), 'CAP > '+str(cap_cut)])
ax.hist([pred_0[cap<=cap_cut],pred_0[cap>cap_cut]], label=['CAP <= '+str(cap_cut), 'CAP > '+str(cap_cut)])
ax.set_xlabel('TEBM probability of progression (dt = 2 years)', fontsize=16)
ax.set_ylabel('Number of people', fontsize=16)
ax.tick_params(labelsize=16)
ax.legend(prop={'size': 16})
print (np.mean(pred_0), np.median(pred_0), np.std(pred_0))
print (len(pred_0[cap<=cap_cut]), len(pred_0[cap>cap_cut]))
print (sp.stats.ttest_ind(pred_0[cap<=cap_cut], pred_0[cap>cap_cut]))

pred_cut = np.mean(pred_0)#0.2298231
fig, ax = plt.subplots()

pred_0 = pred_0[cap>0]
stages_model_bl = stages_model_bl[cap>0]
cap = cap[cap>0]

ax.hist([cap[pred_0<=pred_cut], cap[pred_0>pred_cut]], label=['TEBM prog <= '+str(round(pred_cut,2)), 'TEBM prog > '+str(round(pred_cut,2))])
ax.set_xlabel('CAP score', fontsize=16)
ax.set_ylabel('Number of people', fontsize=16)
ax.tick_params(labelsize=16)
ax.legend(prop={'size': 16}, loc='upper right')
#ax.plot([np.nanmean(cap[pred_0<=pred_cut]),np.nanmean(cap[pred_0<=pred_cut])], [205,208], color='black', linewidth=3)
#ax.plot([np.nanmean(cap[pred_0>pred_cut]),np.nanmean(cap[pred_0>pred_cut])], [205,208], color='black', linewidth=3)
#ax.plot([np.nanmean(cap[pred_0<=pred_cut]),np.nanmean(cap[pred_0>pred_cut])],[208,208], color='black', linewidth=3)
#ax.text((np.nanmean(cap[pred_0>pred_cut])+np.nanmean(cap[pred_0<=pred_cut]))/2-16, 210, '***', fontsize=16)
ax.plot([np.nanmean(cap[pred_0<=pred_cut]),np.nanmean(cap[pred_0<=pred_cut])], [125,128], color='black', linewidth=3)
ax.plot([np.nanmean(cap[pred_0>pred_cut]),np.nanmean(cap[pred_0>pred_cut])], [125,128], color='black', linewidth=3)
ax.plot([np.nanmean(cap[pred_0<=pred_cut]),np.nanmean(cap[pred_0>pred_cut])],[128,128], color='black', linewidth=3)
ax.text((np.nanmean(cap[pred_0>pred_cut])+np.nanmean(cap[pred_0<=pred_cut]))/2-16, 130, '***', fontsize=16)
#ax.set_ylim(0,60)
print (np.mean(cap[pred_0<=pred_cut]), np.mean(cap[pred_0>pred_cut]))
print (sp.stats.ttest_ind(cap[pred_0<=pred_cut], cap[pred_0>pred_cut], nan_policy='omit'))

#plt.show()

"""
fig, ax = plt.subplots()
xspace = np.linspace(0,2,22)
#xspace += [np.random.rand()*2 for i in range(len(xspace))]
xplot0, yplot0 = [], []
xplot1, yplot1 = [], []
count = 1
for delta_t in xspace:
    if delta_t==0:
        continue
    stages_model, _ = model.predict(X, lengths, jumps)
    pred_0, pred_1 = tebm_preds(model, X, lengths, jumps, stages_model, 1, obs_type, scale*delta_t)
#    pred_0 = pred_0[~np.isnan(cag)]
#    cag = cag[~np.isnan(cag)]

    cap = np.array([(x-33.66)*y for x,y in zip(cag,age)])

    #cag_cut = 42
    cap_cut = 400
    print (pred_0[cap<=cap_cut].shape, pred_0[cap>cap_cut].shape)

    xplot0.append(np.array([delta_t]*len(pred_0[cap<=cap_cut])))
    xplot1.append(np.array([delta_t]*len(pred_0[cap>cap_cut])))
    
    if count==1:
        ax.scatter(np.array([delta_t]*len(pred_0[cap<=cap_cut])), pred_0[cap<=cap_cut], color=colors[0], label='CAP <= '+str(cap_cut))
        ax.scatter(np.array([delta_t]*len(pred_0[cap>cap_cut])), pred_0[cap>cap_cut], color=colors[1], label='CAP > '+str(cap_cut))
        yplot1.append(pred_0[cap>cap_cut])
        yplot0.append(pred_0[cap<=cap_cut])
    else:
        ax.scatter(np.array([delta_t]*len(pred_0[cap<=cap_cut])), pred_0[cap<=cap_cut], color=colors[0])
        ax.scatter(np.array([delta_t]*len(pred_0[cap>cap_cut])), pred_0[cap>cap_cut], color=colors[1])
        yplot1.append(pred_0[cap>cap_cut])
        yplot0.append(pred_0[cap<=cap_cut])
    ax.set_ylabel('TEBM progression risk', fontsize=16)
    ax.set_xlabel('Time (years)', fontsize=16)
    #    ax.set_xlim(0,2)
#    ax.set_ylim(0,0.7)
    ax.tick_params(labelsize=16)
    ax.legend(prop={'size': 16})
    count += 1
fig, ax = plt.subplots()
h2d0 = ax.hist2d(np.array(xplot0).flatten(), np.array(yplot0).flatten(), bins=20)
fig.colorbar(h2d0[3])
fig, ax = plt.subplots()
h2d1 = ax.hist2d(np.array(xplot1).flatten(), np.array(yplot1).flatten(), bins=20)
fig.colorbar(h2d1[3])
fig, ax = plt.subplots()
im = ax.imshow(h2d1[0]/h2d0[0], interpolation='nearest', cmap='jet')
ax.set_yticklabels(h2d1[2])
ax.set_xticklabels(h2d1[1])
fig.colorbar(im)
fig, ax = plt.subplots()
im = ax.imshow([[0,2],[1,0]], interpolation='nearest', cmap='jet')
fig.colorbar(im)
print (h2d1)
print (h2d0)
print (h2d1[0]/h2d0[0])
"""
fig, ax = plt.subplots()
from matplotlib import colors as clrs
print (sojourns)
temp = np.array([str(round(x,1)) for x in sojourns])
temp = np.insert(temp, 0, '')
temp = np.insert(temp, 0, '')
print (temp)
hexbin = ax.hist2d(stages_model_predict_bl, pred_0, bins=(np.arange(-0.5, X_predict.shape[1]+1.5, 1),100), cmap=plt.cm.viridis, norm=clrs.LogNorm())
ax.set_xlabel('Disease time (years)', fontsize=16)
ax.set_ylabel('Probability of progression', fontsize=16, labelpad=5)
ax.tick_params(labelsize=16)
ax.set_xticklabels(temp)
ax_t = ax.secondary_xaxis('top')
ax_t.tick_params(axis='x', direction='inout')
ax_t.set_xticklabels(np.arange(X_predict.shape[1]+1))
ax_t.set_xlabel('Stage', fontsize=16)
ax_t.tick_params(labelsize=16)

plt.axhline(pred_cut, color='red', linewidth=3, linestyle='--')

#plt.axhline(pred_cut, xmin=0, xmax=2.75/ax.get_xlim()[1], color='red', linewidth=3, linestyle='--')
#plt.axvline(2.5, ymin=pred_cut/ax.get_ylim()[1], ymax=1.1, color='red', linewidth=3, linestyle='--')
cbar = fig.colorbar(hexbin[3])
cbar.ax.set_yticklabels(['1','10'])
cbar.ax.set_ylabel('Number of people', rotation=270, fontsize=16, labelpad=15)

plt.show()

fig, ax = plt.subplots()
hist_data, labels_data = [], []
stages_model_bl = []
for i in range(len(lengths)):
    nobs_i = lengths[i]
    s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+nobs_i)
    stages_model_bl.append(stages_model[s_idx])
    if stages_model[s_idx] != stages_model[e_idx-1]:
        print (stages_model[s_idx], stages_model[e_idx-1])
stages_model_bl = np.array(stages_model_bl)
for i in range(len(np.unique(labels))):
    hist_data.append(stages_model_bl[labels==np.unique(labels)[i]])
    if np.unique(labels)[i]==1:
        labels_data.append('HD')
    elif np.unique(labels)[i]==2:
        labels_data.append('preHD')
    else:
        labels_data.append('No label')
ax.hist(hist_data, label=labels_data, stacked=True, bins=np.arange(-0.5, X.shape[1]+1.5, 1))
ax.set_xlabel('Stage', fontsize=16)
ax.set_ylabel('Number of individuals', fontsize=16)
ax.tick_params(labelsize=16)
ax.legend(prop={'size': 16})

print ('mean age at stage = 0', np.mean(age[stages_model_bl==0]), np.std(age[stages_model_bl==0]))

labels_long, stages_model_bl_long, pred_0_long, pred_1_long, ids_long = [], [], [], [], []
for i in range(len(lengths)):
    nobs_i = lengths[i]
    for j in range(nobs_i):
        labels_long.append(labels[i])
        stages_model_bl_long.append(stages_model_bl[i])
        pred_0_long.append(pred_0[i])
        pred_1_long.append(pred_1[i])
        ids_long.append(ids[i])

df_data = np.vstack((labels_long,pred_0_long))
df_data = np.vstack((df_data,pred_1_long))
df_data = np.vstack((df_data,X.T[2]))
df_data = np.vstack((df_data,times))
df_data = np.vstack((df_data,ids_long))
df_data = np.vstack((df_data,X.T[0]))
df_data = np.vstack((df_data,stages_model_bl_long))
df_data = np.vstack((df_data,X.T[3]))
df_data = np.vstack((df_data,X.T[4]))
df_data = np.vstack((df_data,X.T[1]))
df_out = pd.DataFrame(data=df_data.T,columns=['DX_bl','PRED_0','PRED_1','TMS','TIME','RID','CAUDATE','STAGE_0','SDMT','TFC','PUTAMEN'])
df_out.to_csv('tebm_trackhd_predictions.csv',index=False)

plt.show()

# TEBM prediction versus MMSE
print ('Single train-test split...')
np.random.seed(42)
# reshape data for train-test split
X_temp, jumps_temp, times_temp, subgroup_temp, tfc_temp = [], [], [], [], []
for i in range(len(lengths)):
    nobs_i = lengths[i]
    X_temp.append(X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    jumps_temp.append(jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    times_temp.append(times[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    subgroup_temp.append(subgroup[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    tfc_temp.append(tfc[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
# shape: n_ppl, n_bms, N_tps
X_temp, jumps_temp, times_temp = np.array(X_temp), np.array(jumps_temp), np.array(times_temp)
X_train, X_test, lengths_train, lengths_test, jumps_train, jumps_test, labels_train, labels_test, times_train, times_test, RID_train, RID_test, X0_train, X0_test, subgroup_train, subgroup_test, tfc_train, tfc_test, dx_change_train, dx_change_test, cag_train, cag_test, age_train, age_test = sklearn.model_selection.train_test_split(X_temp, lengths, jumps_temp, labels, times_temp, ids, X0, subgroup_temp, tfc_temp, dx_change, cag, age, test_size=.5)
# now reshape data back to long format: n_ppl*N_tps, n_bms
X_temp, jumps_temp, times_temp, subgroup_temp, tfc_temp = [], [], [], [], []
for j in range(len(X_train)):
    assert(len(X_train[j])==len(jumps_train[j]))
    for k in range(len(X_train[j])):
        X_temp.append(X_train[j][k])
        jumps_temp.append(jumps_train[j][k])
        times_temp.append(times_train[j][k])
        subgroup_temp.append(subgroup_train[j][k])
        tfc_temp.append(tfc_train[j][k])
X_train, jumps_train, times_train, subgroup_train, tfc_train = np.array(X_temp), np.array(jumps_temp), np.array(times_temp), np.array(subgroup_temp), np.array(tfc_train)
# now reshape data back to long format: n_ppl*N_tps, n_bms
X_temp, jumps_temp, times_temp, subgroup_temp, tfc_temp = [], [], [], [], []
for j in range(len(X_test)):
    assert(len(X_test[j])==len(jumps_test[j]))
    for k in range(len(X_test[j])):
        X_temp.append(X_test[j][k])
        jumps_temp.append(jumps_test[j][k])
        times_temp.append(times_test[j][k])
        subgroup_temp.append(subgroup_test[j][k])
        tfc_temp.append(tfc_test[j][k])
X_test, jumps_test, times_test, subgroup_test, tfc_test = np.array(X_temp), np.array(jumps_temp), np.array(times_temp), np.array(subgroup_test), np.array(tfc_test)
# refit model
#transmat_prior = get_transmat_prior(n_components, order=len(biom_labels), fwd_only=True)
model = tebm_var.MixtureTEBM(X=X_train, lengths=lengths_train, jumps=jumps_train,
                             n_components=n_components, time_mean=1,
                             covariance_type="diag",
                             n_iter=n_iter_inner_em, tol=tol,
                             transmat_prior=transmat_prior, init_params=init_params, params=fit_params,
                             algorithm=algorithm, verbose=False, allow_nan=True,
                             fwd_only=fwd_only, order=order)
seq_model, mixtures = model._fit_tebm(labels_train, n_start=n_start, n_iter=n_iter_seq_em, n_cores=n_cores, model_type=model_type, constrained=True, cut_controls=True)
###################################################
#seq_model = np.array([[8,4,3,0,5,9,7,2,6,1]])
###################################################
print ('Model sequence',seq_model[0],biom_labels[seq_model[0].astype(int)])

# remove controls
def cut_con(X, jumps, lengths, subgroup, tfc, times, ids, labels, dx_change, X0, cag, age):
    X_temp, lengths_temp, jumps_temp, labels_temp, subgroup_temp, tfc_temp, dx_change_temp, times_temp, ids_temp, X0_temp, cag_temp, age_temp = [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(len(lengths)):
        if labels[i] != 0:
            nobs_i = lengths[i]
            for x in X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
                X_temp.append(x)
            for x in jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
                jumps_temp.append(x)
            for x in subgroup[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
                subgroup_temp.append(x)
            for x in tfc[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
                tfc_temp.append(x)
            for x in times[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]:
                times_temp.append(x)            
            ids_temp.append(ids[i])
            lengths_temp.append(lengths[i])
            labels_temp.append(labels[i])
            dx_change_temp.append(dx_change[i])
            X0_temp.append(X0[i])
            cag_temp.append(cag[i])
            age_temp.append(age[i])
    X = np.array(X_temp)
    lengths = np.array(lengths_temp)
    jumps = np.array(jumps_temp)
    labels = np.array(labels_temp)
    subgroup = np.array(subgroup_temp)
    tfc = np.array(tfc_temp)
    dx_change = np.array(dx_change_temp)
    times = np.array(times_temp)
    ids = np.array(ids_temp)
    X0 = np.array(X0_temp)
    cag = np.array(cag_temp)
    age = np.array(age_temp)
    return X, jumps, lengths, labels, subgroup, tfc, dx_change, times, ids, X0, cag, age

print ('Cutting controls!')
X_train, jumps_train, lengths_train, labels_train, subgroup_train, tfc_train, dx_change_train, times_train, ids_train, X0_train, cag_train, age_train = cut_con(X_train, jumps_train, lengths_train, subgroup_train, tfc_train, times_train, RID_train, labels_train, dx_change_train, X0_train, cag_train, age_train)
X_test, jumps_test, lengths_test, labels_test, subgroup_test, tfc_test, dx_change_test, times_test, ids_test, X0_test, cag_test, age_test = cut_con(X_test, jumps_test, lengths_test, subgroup_test, tfc_test, times_test, RID_test, labels_test, dx_change_test, X0_test, cag_test, age_test)

if obs_type == 'Fix':
    stages_model_test, _ = model.predict(X_test, lengths_test)
else:
    stages_model_test, _ = model.predict(X_test, lengths_test, jumps_test)
# baseline stages
stages_model_bl_test = []
for i in range(len(lengths_test)):
    stages_model_bl_test.append(stages_model_test[np.sum(lengths_test[:i])])
stages_model_bl_test = np.array(stages_model_bl_test)

cap_test = np.array([(x-33.66)*y for x,y in zip(cag_test,age_test)])
fig, ax = plt.subplots()
ax.hist(cap_test)
delta_t = 1
pred_0, pred_1 = tebm_preds(model, X_test, lengths_test, jumps_test, stages_model_test, 1, obs_type, scale*delta_t)
pred_0 = pred_0[~np.isnan(cag_test)]
#cag_cut = 42
cap_cut = 400
print (pred_0[cap_test<=cap_cut].shape, pred_0[cap_test>cap_cut].shape)
fig, ax = plt.subplots()
ax.hist([pred_0[cap_test<=cap_cut],pred_0[cap_test>cap_cut]], stacked=True, label=['CAP <= '+str(cap_cut), 'CAP > '+str(cap_cut)])
ax.set_xlabel('TEBM probability of progression (dt = 1 year)', fontsize=16)
ax.set_ylabel('Number of people', fontsize=16)
ax.tick_params(labelsize=16)
ax.legend(prop={'size': 16})
print (np.mean(pred_0), np.median(pred_0), np.std(pred_0))
print (len(pred_0[cap_test<=cap_cut]), len(pred_0[cap_test>cap_cut]))
print (sp.stats.ttest_ind(pred_0[cap_test<=cap_cut], pred_0[cap_test>cap_cut]))

#plt.show()
"""
transmat_boot_arr, Q_boot_arr = [], []
for b in range(1000):
    print ('boot',b+1)
    # generate bootstrap sample
    X_boot, jumps_boot, labels_boot, lengths_boot, k_boot = [], [], [], [], []
    if False:
        # simulated bootstrap
        for i in range(len(lengths)):
            X_i, k_i, dt_i, label_i = model.gen_sample(n_samples=lengths[i], scale=scale)
            lengths_boot.append(lengths[i])
            for j in range(len(X_i)):
                X_boot.append(X_i[j])
                k_boot.append(k_i[j])
                jumps_boot.append(dt_i[j])
            labels_boot.append(label_i)
        X_boot = np.array(X_boot)
        jumps_boot = np.array(jumps_boot)
        labels_boot = np.array(labels_boot)
        lengths_boot = np.array(lengths_boot)
        k_boot = np.array(k_boot)
    else:
        # data bootstrap
        for i in range(len(lengths_train)):
            idx = np.random.randint(len(lengths_train))
            nobs_idx = lengths_train[idx]
            lengths_boot.append(nobs_idx)
            labels_boot.append(labels_train[idx])
            s_idx, e_idx = int(np.sum(lengths_train[:idx])), int(np.sum(lengths_train[:idx])+nobs_idx)
            X_idx = X_train[s_idx:e_idx]
            jumps_idx = jumps_train[s_idx:e_idx]
            for j in range(len(X_idx)):
                X_boot.append(X_idx[j])
                jumps_boot.append(jumps_idx[j])
        X_boot = np.array(X_boot)
        lengths_boot = np.array(lengths_boot)
        labels_boot = np.array(labels_boot)
        jumps_boot = np.array(jumps_boot)

        #        X0_boot = []
        #        for i in range(len(lengths)):
        #            X0_boot.append(X_boot[np.sum(lengths[:i])])
        #        X0_boot = np.array(X0_boot)
        #        mixtures_boot = fit_all_gmm_models(X0_boot, labels_boot)

    if False:#plot_all_data:
        n_x = np.round(np.sqrt(n_bms)).astype(int)
        n_y = np.ceil(np.sqrt(n_bms)).astype(int)
        fig, ax = plt.subplots(n_y, n_x, figsize=(10, 10))
        for i in range(n_bms):
            for j in range(len(lengths)):
                nobs_i = lengths[j]
                s_idx, e_idx = int(np.sum(lengths[:j])), int(np.sum(lengths[:j])+nobs_i)
                #if i==0:
                #    print (jumps_boot[s_idx:e_idx],np.cumsum(jumps_boot[s_idx:e_idx]),X_boot[s_idx:e_idx,i])
                ax[i // n_x, i % n_x].plot(np.cumsum(jumps_boot[s_idx:e_idx]),X_boot[s_idx:e_idx,i])
                ax[i // n_x, i % n_x].set_title(biom_labels[i])
    if False:
        # biomarker distributions and mixture model fits
        #        fig, ax = plotting.mixture_model_grid(X0_boot, np.ones(len(X0_boot)), mixtures, biom_labels)
        #        fig, ax = plotting.mixture_model_grid(X0_boot, labels_boot, mixtures_boot, biom_labels)
        for i in range(X_boot.shape[1]):
            print (biom_labels[i], sp.stats.spearmanr(jumps_boot, X_boot[:,i], nan_policy='omit'))
        print (np.mean(X0_boot, axis=0))

    # uniform prior on prior; minimum transition time should be q_ii=1, maximum is up to user
    # could also investigate Dirichlet prior on Q
    #    time_mean_rnd = np.random.rand()*.4+.1 # min 6 months -> max 30 months
    #    print ('time_mean_rnd',time_mean_rnd)
    model_boot = tebm_var.MixtureTEBM(
        X=X_boot,
        lengths=lengths_boot,
        jumps=jumps_boot,
        n_components=n_components,
        #        time_mean=time_mean_rnd,
        time_mean=1,
        covariance_type="diag",
        n_iter=n_iter_inner_em,
        tol=tol,
        transmat_prior=transmat_prior,
        init_params=init_params,
        params=fit_params,
        algorithm=algorithm,
        verbose=True,
        allow_nan=True,
        fwd_only=fwd_only,
        order=order)
    model_boot.S = seq_model[0]
    # use ML model parameters
    model_boot.prob_mat = get_prob_mat(X_boot, mixtures)
    model_boot.mixtures = mixtures
    model_boot.fit()
    if obs_type == 'Fix':
        transmat_boot = model_boot.a_mat
    else:
        transmat_boot = sp.linalg.expm(scale*np.real(model_boot.Q_))
        Q_boot_arr.append(np.real(model_boot.Q_))
    transmat_boot_arr.append(transmat_boot)

Q_boot_arr = np.array(Q_boot_arr)
transmat_boot_arr = np.array(transmat_boot_arr)

save_variables = {}
save_variables["X"] = X
save_variables["lengths"] = lengths
save_variables["jumps"] = jumps
save_variables["labels"] = labels
save_variables["seq_model"] = seq_model
save_variables["startprob"] = startprob
save_variables["transmat"] = transmat
if obs_type=='Var':
    save_variables["Q"] = model.Q_
save_variables["transmat_boot_arr"] = transmat_boot_arr
save_variables["Q_boot_arr"] = Q_boot_arr
pickle_file = open('./train_1000boots_'+data_out, 'wb')
pickle_output = pickle.dump(save_variables, pickle_file)
pickle_file.close()
"""
pickle_file = open('./test_1000boots_'+data_out, 'rb')
load_variables = pickle.load(pickle_file)
transmat_boot_arr = np.array(load_variables["transmat_boot_arr"])
Q_boot_arr = np.array(load_variables["Q_boot_arr"])
pickle_file.close()

transmat_mean = np.mean(transmat_boot_arr, axis=0)
transmat_std = np.std(transmat_boot_arr, axis=0)
Q_boot_mean = np.mean(Q_boot_arr, axis=0)
Q_boot_std = np.std(Q_boot_arr, axis=0)

fit, ax = plt.subplots()
temp = []
for i in range(len(Q_boot_arr)):
    temp.append(np.sum([-1/x for x in Q_boot_arr[i].diagonal()[1:-1]]))
ax.hist(temp)
ax.set_xlabel('Bootstrapped total durations')

#  sojourn times
sojourns, sojourns_err = [], []
sojourns.append(0)
for i in range(1,len(transmat)-1):     # skip first stage and final stage (absorbing)
    if obs_type=='Var':
        #        sojourn_i = -1/model.Q_[i,i].astype(float)/scale
        sojourn_i = -1/Q_boot_mean[i,i].astype(float)/scale
        # relative error
        sojourn_i_err = -Q_boot_std[i,i]/Q_boot_mean[i,i]
    else:
        sojourn_i = 1/(1-model.a_mat[i,i])
        # relative error
        sojourn_i_err = transmat_std[i,i]/model.a_mat[i,i]
    sojourns.append(sojourn_i)
    sojourns_err.append(sojourn_i_err)
    print ('Stage',i,'duration',round(sojourns[i],2),'+/-',round(sojourns[i]*sojourn_i_err,2),
           'total time so far',round(np.sum(sojourns),2))
print ('Total sequence duration',np.sum(sojourns),'+/-',np.sum(sojourns)*np.sqrt(np.mean(np.power(sojourns_err,2))))
# calculate CIs
confints_x, confints_x_cum = [], []
confints_x.append([0,0])
confints_x_cum.append([0,0])
# percentile bootstrap
for i in range(1,Q_boot_arr.shape[1]-1):
    Q_vec = Q_boot_arr[:,i,i].astype(float)
    Q_mean = model.Q_[i,i].astype(float)
    d_i = np.array([np.percentile(Q_vec, 97.5), np.percentile(Q_vec, 2.5)]) # [upper, lower]
    Q_upper = d_i[0]
    Q_lower = d_i[1]
    x_mean = -1/Q_mean/scale
    x_upper = -1/d_i[0]/scale
    x_lower = -1/d_i[1]/scale
    confints_x.append([x_mean-x_lower,x_upper-x_mean]) # [-,+]
    # cumulative uncertainty
    confints_x_cum.append(np.sqrt(np.sum(np.power(confints_x,2),axis=0)))
"""
#FIXME
# empirical bootstrap
for i in range(1,Q_boot_arr.shape[1]-1):
    Q_vec = Q_boot_arr[:,i,i].astype(float)
    Q_mean = model.Q_[i,i].astype(float)
    Q_vec -= Q_mean
    d_i = np.array([np.percentile(Q_vec, 97.5), np.percentile(Q_vec, 2.5)]) # [upper, lower]
    Q_upper = Q_mean - d_i[1]
    Q_lower = Q_mean - d_i[0]
    print (d_i, Q_mean)
    print (Q_lower, Q_mean, Q_upper)
    x_mean = -1/Q_mean/scale
    x_upper = -1/Q_upper/scale
    x_lower = -1/Q_lower/scale
    print (x_lower, x_mean, x_upper)
    confints_x.append([x_mean-x_lower,x_upper-x_mean]) # [-,+]
    # cumulative uncertainty
    confints_x_cum.append(np.sqrt(np.sum(np.power(confints_x,2),axis=0)))
"""
#quit()
confints_x = np.array(confints_x)
# cumulative error = RMS of all previous errors
#confints_x_cum = np.array(confints_x_cum)
# cumulative error = sum of all previous errors
confints_x_cum = np.cumsum(confints_x, axis=0)
# dummy errors on y; could make this equal to event sequence uncertainty
confints_y = np.ones(confints_x.shape)*.5
confints_y[0] = 0
#
fig, ax = plt.subplots()
sojourns_boot = []
for i in range(Q_boot_arr.shape[1]-1):
    sojourns_boot.append([-1/x/scale for x in Q_boot_arr[:,i,i]])
    if i>0:
        sojourns_boot[i] = [x+np.mean(sojourns_boot[i-1]) for x in sojourns_boot[i]]
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
              for x, y, xe, ye in zip(np.cumsum(sojourns), np.arange(len(sojourns)), confints_x_cum, confints_y)]
colours = [colors[0],colors[1],colors[1],colors[2],colors[1],colors[1],colors[2],colors[2],colors[1],colors[1],colors[1],colors[1]]
pc = PatchCollection(errorboxes, facecolor=colours, alpha=0.5, edgecolor='none')
ax.add_collection(pc)
ax.scatter(np.cumsum(sojourns), np.arange(len(sojourns)), c='black', s=20)
ax.errorbar(np.cumsum(sojourns), np.arange(len(sojourns)), xerr=confints_x.T, capsize=5, capthick=2, ls='none', c='black')
ax.set_yticks(np.arange(len(sojourns)))
#ax.set_yticklabels(np.array(biom_labels)[seq_model[0].astype(int)], ha='right', rotation_mode='anchor')
ax.set_yticklabels(np.array(event_labels)[seq_model[0].astype(int)], ha='right', rotation_mode='anchor')
ax.set_xlabel('Time (years)', fontsize=16)
ax.set_xlim(0,30)
ax.tick_params(labelsize=16)
ax.grid(True)
ax.set_title('Timeline inferred from TrackHD', fontsize=16)
#plt.show()
"""
for i in range(len(transmat)-1):     # skip final stage (absorbing)
    if obs_type=='Var':
        sojourn_i = -1/model.Q_[i,i].astype(float)/scale
    else:
        sojourn_i = 1/(1-self.transmat_[i,i])
    sojourns.append(sojourn_i)
    print ('Stage',i,'duration',round(sojourns[i+1],2),'total time so far',round(np.sum(sojourns),2))
print ('Total sequence duration',np.sum(sojourns))
"""

# staging
sojourns = np.cumsum(sojourns)
if obs_type == 'Fix':
    stages_model, _ = model.predict(X, lengths)
else:
    stages_model, _ = model.predict(X, lengths, jumps)
    stages_model_prob = model.predict_proba(X, lengths, jumps)

"""
fig, ax = plt.subplots()
transmat_weighted = np.zeros(transmat.shape)
for x in stages_model_prob:
    row_sums = (transmat*x).sum(axis=1)
    x_norm = (transmat*x) / row_sums[:, np.newaxis]
    transmat_weighted += x_norm
ax.imshow(transmat_weighted, interpolation='nearest', cmap=plt.cm.Reds)
"""
probprog = []
for i in range(len(lengths)):
    nobs_i = lengths[i]
    s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+nobs_i)
    stages_i = stages_model[s_idx:e_idx]
    post_i = stages_model_prob[s_idx:e_idx]
    for j in range(nobs_i):
        pred_ij = np.dot(post_i[j], transmat)
        #        pred_ij = np.dot(post_i[j], sp.linalg.expm(3*np.real(model.Q_)))
        #        probprog.append(1-pred_ij[stages_i[j]])
        #        probprog.append(1-transmat[np.argmax(pred_ij),np.argmax(pred_ij)])
        #        probprog.append(1-transmat[stages_i[j],stages_i[j]])
        #        probprog.append(1-sp.linalg.expm(np.real(model.Q_))[stages_i[j],-1])
        #        probprog.append(np.argmax(pred_ij)-stages_i[j])
        probprog.append(np.average(np.arange(X.shape[1]+1), weights=pred_ij) - np.average(np.arange(X.shape[1]+1), weights=post_i[j]))
probprog = np.array(probprog)

# baseline stages
stages_model_bl, delta_tms, subgroup_bl, probprog_bl, delta_subgroup = [], [], [], [], []
for i in range(len(lengths)):
    stages_model_bl.append(stages_model[np.sum(lengths[:i])])
    subgroup_bl.append(subgroup[np.sum(lengths[:i])])   
    probprog_bl.append(probprog[np.sum(lengths[:i])])
    nobs_i = lengths[i]
    s_idx, e_idx = int(np.sum(lengths[:i])), int(np.sum(lengths[:i])+nobs_i)
    delta_tms.append(X[e_idx-1,5]-X[s_idx,5])
    delta_subgroup.append(subgroup[e_idx-1]-subgroup[s_idx])
stages_model_bl = np.array(stages_model_bl)
#print (np.unique(delta_subgroup, return_counts=True))

print (stages)
print (ids)
        
fig, ax = plt.subplots()
hist_data, labels_data = [], []
for i in range(len(np.unique(labels))):
    hist_data.append(stages_model_bl[labels==np.unique(labels)[i]])
    labels_data.append(str(np.unique(labels)[i]))
ax.hist(hist_data, label=labels_data, stacked=True, bins=np.arange(-0.5, X.shape[1]+1.5, 1))
ax.legend()

df_data = np.vstack((stages_model,subgroup))
df_data = np.vstack((df_data,probprog))
df_data = np.vstack((df_data,X[:,5]))
#df_data = np.vstack((df_data,delta_tms))
#df_data = np.vstack((df_data,dx_change))
#df_data = np.vstack((df_data,delta_subgroup))
#df_out = pd.DataFrame(data=df_data.T,columns=['STAGE','SUBGROUP','PROBPROG','TMS','DXCHANGE','DSUBGROUP'])
df_out = pd.DataFrame(data=df_data.T,columns=['STAGE','SUBGROUP','PROBPROG','TMS'])
df_out.to_csv('tebm_trackhd_preds.csv',index=False)

plt.show()

# timeline
age = data[data['TIME']==0]['age'].values
#plot_timeline(X, lengths, jumps, labels, event_labels, model, transmat, sojourns, stages_model, obs_type)
# survival probabilities by stage
if obs_type=='Var':
    colors = ['C{}'.format(x) for x in range(10)]
    colors.append('black')
    colors.append('grey')
    styles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,10)), (0,(1,1)), (0,(5,10)), (0,(5,5)), (0,(5,1)), (0,(3,10,1,10)), (0,(3,5,1,5)), (0,(3,1,1,1))]
    fig, ax = plt.subplots()
    xspace = np.linspace(0,20,100)
    for i in range(model.Q_.shape[0]-1):
    #    for i in range(8):
        yspace_i = []
        for j in range(len(xspace)):
            # survival from state i to final state
            yspace_i.append(1-sp.linalg.expm(xspace[j]*np.real(model.Q_))[i,-1])
        ax.plot(xspace/scale, yspace_i, label='Stage '+str(i), color=colors[i], linestyle=styles[i])
    ax.set_xlabel('Time (years)', fontsize=16, labelpad=5)
    ax.set_ylabel('Survival probability', fontsize=16, labelpad=5)
    ax.tick_params(labelsize=16)
    plt.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01), prop={'size': 14})
    plt.subplots_adjust(top=0.99, right=0.99, left=.1, bottom=.1)
# example trajectories
#plot_examples(X, lengths, jumps, labels, model, transmat, sojourns, stages_model, age, obs_type)
# all trajectories
fig, ax = plt.subplots()
for i in range(len(lengths)):
    nobs_i = lengths[i]
    stages_i = stages_model[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
    jumps_i = jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
    times_i = times[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
    ax.plot(times_i+np.random.normal(loc=0,scale=0.1), stages_i+np.random.normal(loc=0,scale=0.1))
ax.set_xlabel('Time')
ax.set_ylabel('Stage')


#old CI code
"""
for i in range(1,Q_boot_arr.shape[1]-1):
    # data to resample
    Q_vec = Q_boot_arr[:,i,i].astype(float)

#    Q_vec = Q_vec[Q_vec<-1E-2]    
#    Q_vec = np.array([-1/x for x in Q_vec])
    
    print (np.mean(Q_vec))
    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='bca')
    print (ci.confidence_interval)
#    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='percentile')
#    print (ci.confidence_interval)
#    ci = bootstrap((Q_vec,), np.mean, confidence_level=0.95, method='basic')
#    print (ci.confidence_interval)

    #    Q_vec = Q_vec[Q_vec<-0.05]
    #    Q_vec = Q_vec[Q_vec>-4]
    
    # bootstrap with replacement samples of same length
#    Q_vec_boot = np.array([np.mean(resample(Q_vec)) for i in range(10000)])
    Q_vec_boot = Q_vec
    # subtract data mean
    Q_mean = np.mean(Q_vec)
#    Q_mean = model.Q_[i,i].astype(float) # could be ML fit mean? don't think so, since it should be the mean of the sample used for resampling

#    fig, ax = plt.subplots()
#    ax.hist(Q_vec_boot)
    
#    Q_vec_boot -= Q_mean

#    fig, ax = plt.subplots()
#    ax.hist(Q_vec_boot)
#    plt.show()
    
    # calculate percentiles
    #    d_i = np.abs(np.array([np.percentile(Q_vec_boot, 2.5), np.percentile(Q_vec_boot, 97.5)])) # [-,+]
    #    d_i = np.array([1.96*np.std(Q_vec)/np.sqrt(len(Q_vec)), 1.96*np.std(Q_vec)/np.sqrt(len(Q_vec))])
    d_i = np.array([ci.confidence_interval[0], ci.confidence_interval[1]])

    
    print ('!',Q_mean, d_i)
#    fig, ax = plt.subplots()
#    ax.hist(Q_vec)
#    ax.axvline(model.Q_[i,i].astype(float))
#    plt.show()
    # convert to unit time
    #    x_upper = -1/(Q_mean-d_i[0])
    #    x_lower = -1/(Q_mean-d_i[1])

    #    x_mean = -1/model.Q_[i,i].astype(float)
    #    x_upper = x_mean*d_i[0]/Q_mean
    #    x_lower = x_mean*d_i[1]/Q_mean

#    mean_log = np.log(-Q_mean)
#    stderr_log = ci.standard_error*1/(-Q_mean)
#    x_mean = np.exp(mean_log)
#    x_upper = np.exp(mean_log + 1.96*stderr_log)
#    x_lower = np.exp(mean_log - 1.96*stderr_log)
#    print (x_lower, x_mean, x_upper)
    
    #    confints_x.append([x_mean-x_lower, x_upper-x_mean]) # [-,+]
    #    confints_x.append([Q_mean-d_i[0], d_i[1]-Q_mean]) # [-,+]
    confints_x.append([-1/model.Q_[i,i].astype(float)*d_i[1]/Q_mean, -1/model.Q_[i,i].astype(float)*d_i[0]/Q_mean]) # [-,+]
    print (confints_x[i])
    # cumulative uncertainty
    confints_x_cum.append(np.sqrt(np.sum(np.power(confints_x,2),axis=0)))
"""
