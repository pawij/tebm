import numpy as np
from matplotlib import pyplot as plt
import sys
import sklearn
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import FormatStrFormatter
import scipy as sp

from tebm import tebm_fix, cthmm_fix, tebm_var
from kde_ebm.mixture_model import get_prob_mat
import sys
sys.path.insert(1, '/home/pwijerat/code/GP_progression_model_V2-develop/lib/python3.7/site-packages/gppm-2.0.0-py3.7.egg')
import GP_progression_model
import torch
import pandas as pd
import pickle

colors = ['C{}'.format(x) for x in range(10)]

def get_transmat_prior(n_components, order=0, fwd_only=False):
    # prior on transition matrix
    transmat_prior = np.ones((n_components,n_components))
    if fwd_only:
        for i in range(len(transmat_prior)):
            transmat_prior[i,i] = 1/n_components
            transmat_prior[i,:i] = 0.
            if (i+order+1) < len(transmat_prior):
                transmat_prior[i,i+order+1:] = 0.
            count_nonzero = np.count_nonzero(transmat_prior[i]!=0)
            # distribute probability to nonzero states
            for j in range(n_components):
                transmat_prior[i,:i] = 0.
                if i!=j and transmat_prior[i,j]!=0.:
                    transmat_prior[i,j] = (1-transmat_prior[i,i])/(count_nonzero-1)
                elif i==(n_components-1) and (j==n_components-1):
                    transmat_prior[i,j] = 1.
    else:
        for i in range(len(transmat_prior)):
            transmat_prior[i] /= np.sum(transmat_prior[i])
    return transmat_prior

def sample_stage_fix(model, stages, lengths, i):
    nobs_i = lengths[i]
    X_sample = []
    for x in range(100):
        for j in range(len(stages)):
            X_sample.append(model._generate_sample_from_state(stages[j]))
    X_sample = np.array(X_sample)
    t_sample, _ = model.predict(X_sample, np.array([nobs_i for x in range(int(len(X_sample)/nobs_i))]))
    t_sample = t_sample.reshape(int(len(X_sample)/nobs_i), nobs_i)
    X_sample = X_sample.reshape(int(len(X_sample)/nobs_i), X_sample.shape[1], nobs_i)
    return t_sample

def sample_stage_var(model, stages, lengths, jumps, i):
    nobs_i = lengths[i]
    X_sample = []
    jumps_local = []
    for x in range(100):
        for j in range(len(stages)):
            X_sample.append(model._generate_sample_from_state(stages[j]))
            jumps_local.append(jumps[j])
    X_sample = np.array(X_sample)
    t_sample, _ = model.predict(X_sample, np.array([nobs_i for x in range(int(len(X_sample)/nobs_i))]), jumps_local)
    t_sample = t_sample.reshape(int(len(X_sample)/nobs_i), nobs_i)
    X_sample = X_sample.reshape(int(len(X_sample)/nobs_i), nobs_i, X_sample.shape[1])
    return t_sample, X_sample

#def crossval(dmmse, dx_change, data, X, lengths, labels, n_components, cut_nans, order):
def crossval(dmmse, all_data_0, all_data_1, all_data_2, X, lengths, labels, n_components, cut_nans, order, is_tebm):
    np.random.seed(0)
    # forward prior
    transmat_prior_fw = np.ones((n_components,n_components))
    for i in range(len(transmat_prior_fw)):
        transmat_prior_fw[i,:i] = 0.
        transmat_prior_fw[i] /= np.sum(transmat_prior_fw[i])
    # forward-backward prior
    transmat_prior_fwbw = np.ones((n_components,n_components))    
    for i in range(len(transmat_prior_fwbw)):
        if (i-order) >= 0:
            transmat_prior_fwbw[i,:(i-order)] = 0.
        if (i+order+1) < len(transmat_prior_fwbw):
            transmat_prior_fwbw[i,i+order+1:] = 0.
        transmat_prior_fwbw[i] /= np.sum(transmat_prior_fwbw[i])
    ### cross-validation
    N_folds = 10
    print ('CV over',N_folds,'folds...')
    test_idxs, train_idxs = [], []
    cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
    # reshape data into 1 row / person for cv.split
    X_temp = []
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        X_temp.append(X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    # shape: N_ppl, N_bms, N_tps
    X_temp = np.array(X_temp)
    for train, test in cv.split(X_temp,np.array(labels)):
        test_idxs.append(test)
        train_idxs.append(train)
    test_idxs = np.array(test_idxs)
    train_idxs = np.array(train_idxs)

    pred_all = []
    
    cv_auc_stage, cv_auc_pred, cv_auc_dstage = [], [], []
    for idx in range(len(test_idxs)):
        lengths_train = lengths[train_idxs[idx]]
        labels_train = labels[train_idxs[idx]]
        # now reshape data back to long format: N_ppl*N_tps, N_bms
        X_train = X_temp[train_idxs[idx]]
        temp = []
        for j in range(len(X_train)):
            for k in range(len(X_train[j])):
                temp.append(X_train[j][k])
        X_train = np.array(temp)
        
        lengths_test = lengths[test_idxs[idx]]
        # now reshape data back to long format: N_ppl*N_tps, N_bms
        X_test = X_temp[test_idxs[idx]]
        temp = []
        for j in range(len(X_test)):
            for k in range(len(X_test[j])):
                temp.append(X_test[j][k])
        X_test = np.array(temp)

        #FIXME: set dx_change externally from initial data read-in
        true_conv = return_labels(all_data_0, all_data_1, all_data_2)[test_idxs[idx]]
        #        true_conv = dx_change[test_idxs[idx]]

        if is_tebm:
            # first fit model on training data with forward-backward prior to get sequence
            model = tebm_fix.MixtureTEBM(X=X_train, lengths=lengths_train, 
                                         n_components=n_components, covariance_type="diag",
                                         n_iter=1, tol=1E-3,
                                         init_params='st', params='st',
                                         verbose=False, allow_nan=True)
            ml_seq, mixtures = model._fit_tebm(labels_train, n_start=1, n_iter=100, n_cores=1, model_type='GMM', cut_controls=False)
            # now refit with forward-only prior
            model = tebm_fix.MixtureTEBM(X=X_train, lengths=lengths_train, 
                                         n_components=n_components, transmat_prior=transmat_prior_fw, covariance_type="diag",
                                         n_iter=1, tol=1E-3,
                                         init_params='s', params='st',
                                         verbose=False, allow_nan=True)
            model.S = ml_seq[0]
            model.mixtures = mixtures
            model.prob_mat = get_prob_mat(X_train, mixtures)
        else:
            max_like = -np.inf
            max_n = 0
            like_arr = []
            for n in range(1, 20):
                print ('n_states',n)
                model = cthmm_fix.GaussianCTHMM(X=X_train, lengths=lengths_train, 
                                                n_components=n, covariance_type="diag",
                                                n_iter=100, tol=1E-3,
                                                init_params='mcst', params='mcst',
                                                verbose=False, allow_nan=False)
                model.fit()
                like = model.score(X_train, lengths_train)
                like_arr.append(like)
                if like > max_like:
                    max_like = like
                    max_n = n
                    max_model = model
                else:
                    break
            print ('ML number of states',max_n,max_like)
            n_components = max_n
            # now refit with forward-only prior
            transmat_prior_fw = np.ones((n_components,n_components))
            for i in range(len(transmat_prior_fw)):
                transmat_prior_fw[i,:i] = 0.
                transmat_prior_fw[i] /= np.sum(transmat_prior_fw[i])            
            model = cthmm_fix.GaussianCTHMM(X=X_train, lengths=lengths_train, 
                                            n_components=n_components, transmat_prior=transmat_prior_fw, covariance_type="diag",
                                            n_iter=100, tol=1E-3,
                                            init_params='mcst', params='mcst',
                                            verbose=False, allow_nan=False)
        model.fit()
        stages_model, _ = model.predict(X_test, lengths_test)
        ### done up to here
        bl_stage_tebm = []
        for i in range(len(lengths_test)):
            bl_stage_tebm.append(stages_model[np.sum(lengths_test[:i])])
        bl_stage_tebm = np.array(bl_stage_tebm)
        true_fast = return_true_conv(dmmse[test_idxs[idx]], lengths_test, time_idx=1, predict_mri=False)
        #        true_fast = return_true_conv(X_test, lengths_test, time_idx=2, predict_mri=True)
        if not cut_nans:
            pred_tebm, stage_tebm = tebm_preds(model, X_test, lengths_test, time_idx=1)
            auc_pred_tebm = roc_auc_score(true_conv, pred_tebm)
        else:
            stage_tebm = bl_stage_tebm
        auc_stage_tebm, _ = calc_roc_mba(bl_stage_tebm, true_conv, n_components, is_tebm=True)
        print ('auc_stage_tebm',auc_stage_tebm)
        cv_auc_stage.append(auc_stage_tebm)
        if not cut_nans:
            print ('auc_pred_tebm',auc_pred_tebm)
            cv_auc_pred.append(auc_pred_tebm)
        
    print (N_folds,'fold CV auc_stage',np.mean(cv_auc_stage),'std',np.std(cv_auc_stage))
    if not cut_nans:
        print (N_folds,'fold CV auc_pred',np.mean(cv_auc_pred),'std',np.std(cv_auc_pred))

def return_labels(data_0, data_1, data_2):
    true_conv = [0]*len(data_0)
    for i,row in data_0.iterrows():
        dx_change_i_1 = data_1[data_1['RID']==row['RID']]['DXCHANGE'].values
        if dx_change_i_1.size == 0:
            dx_change_i_1 = np.nan
        dx_change_i_2 = data_2[data_2['RID']==row['RID']]['DXCHANGE'].values
        if dx_change_i_2.size == 0:
            dx_change_i_2 = np.nan
        if (4 == dx_change_i_1 or 5 == dx_change_i_1) or (4 == dx_change_i_2 or 5 == dx_change_i_2):
            true_conv[i] = 1
    true_conv = np.array(true_conv)
    return true_conv

def calc_roc_mba(stages, true_conv, n_components, is_tebm):
    #    print('Total n_conv',sum(true_conv),'n_stbl',len(true_conv)-sum(true_conv))
    mba_stg = 0
    max_bac = 0.
    spec_arr = []
    sens_arr = []
    if is_tebm:
        j_max = n_components
    else:
        j_max = n_components+1
    for j in range(j_max):
        pred_conv = [1]*len(stages)
        for k in range(len(stages)):# stage at baseline
            if int(stages[k]) < j:
                pred_conv[k] = 0
                # sanity check
                #        if sum(pred_stbl)+sum(pred_conv)!=sum(true_conv)+sum(true_stbl):
                #            print(sum(pred_stbl),sum(pred_conv),sum(true_conv),sum(true_stbl))
                #            quit()
        spec = 0.
        sens = 0.
        # assumes that indices are consistent
        for k in range(len(true_conv)):
            if (true_conv[k] == 1 and pred_conv[k] == 1):
                spec+=1.
            elif (true_conv[k] == 0 and pred_conv[k] == 0):
                sens+=1.
        spec /= float(sum(true_conv))
        sens /= float(len(true_conv)-sum(true_conv))
        spec_arr.append(spec)
        sens_arr.append(sens)
#        print (j, spec, sens)
        if (spec+sens)/2.0 > max_bac:
            max_bac = (spec+sens)/2.0
            mba_stg = j
            spec_max = spec
            sens_max = sens
    tpr_arr = np.array(spec_arr)[::-1]
    fpr_arr = np.array([1.-x for x in sens_arr])[::-1]
    fpr_diff = np.diff(fpr_arr)
    roc = 0.
    # trapezoidal integration
    for i in range(len(fpr_diff)):
        roc += .5*fpr_diff[i]*(tpr_arr[i]+tpr_arr[i+1])
    #    print('mba',max_bac,'spec',spec_max,'sens',sens_max,'mba stage',mba_stg)
    #    print('roc',roc)
    #    fig, ax = plt.subplots()
    #    ax.plot(fpr_arr, tpr_arr)
    #    plt.show()
    #    return max_bac
    return roc, max_bac

def return_true_conv(X_temp, lengths_temp, time_idx, predict_mri=True):
    true_conv_temp = np.zeros(len(lengths_temp))
    for i in range(len(lengths_temp)):
        nobs_i = lengths_temp[i]
        if predict_mri:
            X_i = X_temp[np.sum(lengths_temp[:i]):np.sum(lengths_temp[:i])+nobs_i]
            if np.mean(X_i[time_idx])/np.mean(X_i[time_idx-1]) > 1.05:
                true_conv_temp[i] = 1
        else:
            if np.abs(X_temp[i]) > 6:
                true_conv_temp[i] = 1
    return true_conv_temp

def tebm_preds(model, X, lengths, jumps, bl_stage, time_idx, obs_type, scale):

    prog_0, prog_1 = np.zeros(len(lengths)), np.zeros(len(lengths))
    stage = np.zeros(len(lengths))
    for i in range(len(lengths)):
        # inferred posterior using number of measurements = time_idx
        if obs_type=='Fix':
            model.X = X[np.sum(lengths[:i]):np.sum(lengths[:i])+time_idx].reshape(time_idx,X.shape[1])
            model.lengths = lengths[i]
            post = model.posteriors_X(X[np.sum(lengths[:i]):np.sum(lengths[:i])+time_idx].reshape(time_idx,X.shape[1]),np.array([lengths[i]]))
        else:
            post = model.predict_proba(X[np.sum(lengths[:i]):np.sum(lengths[:i])+time_idx].reshape(time_idx,X.shape[1]),
                                       np.array([1]),
                                       np.array([0]))
        # prevent prob = 0
        post[post==0] = sys.float_info.min
        # posterior at reference time-point
        post = post[time_idx-1]
        # max like stage
        ############################################################################## NB: stage[i] not necessarily = argmax(post), because only 1 tp used
        #        stage[i] = np.argmax(post)        
        stage[i] = bl_stage[i]
        # predicted emissions
        if obs_type=='Fix':
            transmat = model.a_mat
        else:
            #FIXME: how to choose scale
            transmat = sp.linalg.expm(scale*np.real(model.Q_))
            #            transmat = sp.linalg.expm(np.sum(jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+lengths[i]])*np.real(model.Q_))
        pred = np.dot(post, transmat)
        # prevent prob = 0
        pred[pred==0] = sys.float_info.min

        ### OLD: mixes staging with progression; non-monotonic
        # prob of progression - corr = 0.366
#        prog_0[i] = 1-np.sum(pred/(pred+post))/pred.shape[0]
        
        xspace = np.arange(len(pred))
        
        ### NEW: simple change in prob of being at max like stage at baseline and after delta time
        # METRIC 1
        prog_0[i] = np.abs(post[np.argmax(post)]-pred[np.argmax(post)])
        # Danny's
        #        prog_0[i] = np.sum(np.abs(post[:np.argmax(post)+1]-pred[:np.argmax(post)+1]))
        # summing all probability >= max baseline stage
#        prog_0[i] = np.sum(np.abs(post[np.argmax(post):]-pred[np.argmax(post):]))
        # skewness (distribution should shift to the right)
#        prog_0[i] = np.abs(sp.stats.skew(post)-sp.stats.skew(pred))
        # METRIC 2
        #        prog_0[i] = (np.abs(post[np.argmax(post)]-pred[np.argmax(post)]) + np.abs(post[np.argmax(pred)]-pred[np.argmax(pred)]))/2
        #        prog_0[i] = (post[np.argmax(post)]-pred[np.argmax(post)])/(post[np.argmax(post)]+pred[np.argmax(post)])
        #        prog_0[i] = post[int(stage[i])]-pred[int(stage[i])]

        ### NEW: difference in probability mass; this is stage-dependent
        #        prog_0[i] = (np.average(xspace, weights=pred) - np.average(xspace, weights=post))#/(len(pred)-1)
        #        prog_0[i] = ((np.average(xspace, weights=pred) - np.average(xspace, weights=post))/np.average(xspace, weights=pred) + np.average(xspace, weights=post))/(len(pred)-1)

        ### NEW: similarity metrics
        #        prog_0[i] = sp.stats.wasserstein_distance(post,pred)
        #        prog_0[i] = np.sum(pred*np.log(pred/post))
        
        if prog_0[i] < 0:
            print (stage[i])
        if False:
            print (stage[i], np.argmax(post), prog_0[i])
            fig, ax = plt.subplots()
            ax.bar(xspace, post)
            fig, ax = plt.subplots()
            ax.bar(xspace, pred)
            fig, ax = plt.subplots()
            ax.bar(xspace, post-pred)
            plt.show()

        """
        fig, ax = plt.subplots()
        ax.bar(xspace, post)
        fig, ax = plt.subplots()
        ax.bar(xspace, pred)
        fig, ax = plt.subplots()
        #        ax.bar(xspace, pred/(post+pred))
        ax.bar(xspace, pred-post)
        temp = pred-post
        print (np.average(xspace, weights=temp))
        plt.show()
        """
        
        # just using prob of transition - corr = 0.263
        prog_1[i] = 1-pred[int(stage[i])]
        # normalised prob of transition - corr = 0.251
        #        prog[i] = 1-pred[int(stage[i])]/(pred[int(stage[i])]+post[int(stage[i])])
        # diagonal forward prob - corr = 0.325
        #        prog_1[i] = 1-np.sum([post[ii]*model.transmat_[ii,ii] for ii in range(len(post))])        
        """
        if prog[i] < 0:
            print ('!!!',pred[int(stage[i])],post[int(stage[i])])
            prog[i] = 0
        elif prog[i] > 1:
            print ('???',pred[int(stage[i])],post[int(stage[i])])
            prog[i] = 1
        """
    return prog_0, prog_1

def pred_traj(model, X, stages, lengths, jumps, idx, sojourns):
    # prediction using actual observation
    post = model.predict_proba(X[idx].reshape(1,X[idx].shape[0]),
                               np.array([1]),
                               np.array([0]))
    # prevent prob = 0
    post[post==0] = sys.float_info.min
    # posterior at reference time-point
    post = post[0]
    pred_stage = []
    stage_prev = stages[0]
    if stage_prev == len(sojourns):
        t_sample, X_sample = sample_stage_var(model, np.array([stage_prev]), [1], [0], 0)
        return np.array([[stage_prev, np.mean(t_sample), np.std(t_sample)]])
    pred_stage.append([stage_prev,sojourns[stage_prev]-sojourns[1]])
    # FIXME: should set time step externally; depends on time resolution used to train 'model'
    for t in np.linspace(1,21,80):# step = 0.25, corresponding to smallest time resolution in ADNI (3 months)
        transmat = sp.linalg.expm(t*np.real(model.Q_))
        pred = np.dot(post, transmat)
        # prevent prob = 0
        pred[pred==0] = sys.float_info.min
        stage_t = np.argmax(pred)
        # find first time of stage change    
        if stage_t > stage_prev:
            pred_stage.append([stage_t,t+pred_stage[0][1]])
            stage_prev = stage_t
    # uncertainty from sampling model at predicted stages
    master_arr = []
    for j in range(len(pred_stage)):
        if pred_stage[j][0]==1:
            time_sample = 0
        else:
            t_sample, X_sample = sample_stage_var(model, np.array([pred_stage[j][0]]), [1], [0], 0)
            time_sample = []
            for i in range(len(t_sample)):
                time_sample.append(sojourns[t_sample[i]-1]) # -1 because sojourns starts from event 1 (i.e. it ignores stage 0, which isn't allowed in this function)
        master_arr.append([pred_stage[j][0], pred_stage[j][1], np.std(time_sample)])
        #        master_arr.append([pred_stage[j][0], np.mean(time_sample), np.std(time_sample)])
    return np.array(master_arr)

def model_likelihood(X, lengths, model):
    like = []
    for i, j in iter_from_X_lengths(X, lengths):
        l = model._compute_log_likelihood(X[i:j])
        for t in range(len(l)):
            like.append(l[t])
    return like
            
def linspace_local2(a, b, N, arange_N):
    return a + (b - a) / (N - 1.) * arange_N

def calc_coeff(sig):
    return 1. / np.sqrt(np.pi * 2.0) * sig

def calc_exp(x, mu, sig):
    x = (x - mu) / sig
    return np.exp(-.5 * x * x)

def plot_timeline(X, lengths, jumps, labels, event_labels, model, transmat, sojourns, stages_model, obs_type):    
    # plot timeline and example patients
    fig, ax = plt.subplots()
    stages_model_bl = []
    for i in range(len(lengths)):
        stages_model_bl.append(stages_model[np.sum(lengths[:i])])
    stages_model_bl = np.array(stages_model_bl)
    got_mci, got_ad = 0, 0
    for i in range(len(stages_model_bl)):
        nobs_i = lengths[i]
        X_i = X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        if obs_type == 'Fix':
            post = model.predict_proba(X_i, np.array([lengths[i]]))
        else:
            jumps_i = jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
            post = model.predict_proba(X_i, np.array([lengths[i]]), jumps_i)
        p_pred = np.dot(post,transmat)
        stage_pred = np.argmax(p_pred, axis=1)
        if labels[i] == 2 and stages_model_bl[i]>0 and (stage_pred[0]-stages_model_bl[i])==1 and not got_mci:
            ax.arrow(0,.25,sojourns[int(stages_model_bl[i])],0,length_includes_head=True,head_width=.1,head_length=.6,shape='full')
            ax.arrow(sojourns[int(stages_model_bl[i])],.25,sojourns[int(stage_pred[0])]-sojourns[int(stages_model_bl[i])],0,length_includes_head=True,head_width=.1,head_length=.6,shape='full',alpha=.5)
            ax.text(0,.31,'Patient X (MCI)',fontsize=15)
            got_mci=1
        if labels[i] == 1 and stages_model_bl[i]>0 and stage_pred[0]>=stages_model_bl[i] and not got_ad:
            ax.arrow(0,.4,sojourns[int(stages_model_bl[i])],0,length_includes_head=True,head_width=.1,head_length=.6,shape='full')
            ax.text(0,.45,'Patient Y (AD)',fontsize=15)
            got_ad=1
    ax.set_xlim(0,np.max(sojourns))
    for i in range(len(sojourns)):
        ax.text(sojourns[i],-.1,str(round(sojourns[i],1)),fontsize=15)
        ax.text(sojourns[i],.05,event_labels[i],rotation=45,fontsize=15)
    ax.set_xticklabels([])
    ax.set_xticks(sojourns)
    ax.tick_params(length=20,labelsize='large')
    ax.set_xlabel('Disease time (years)',fontsize=20,labelpad=40)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.subplots_adjust(top=0.99, left=0.05, bottom=.2)

def plot_examples(X, lengths, jumps, labels, model, transmat, sojourns, stages_model, age, obs_type):
    # select and plot 3 example trajectories
    done_0, done_1, done_2 = 0, 0, 0
    count, dist = 0, 0.
    fsize = 25
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        stages_i = stages_model[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        X_i = X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        time_i = [sojourns[j] for j in stages_i]
        jumps_i = jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        #        if (time_i[0]==time_i[-1]) and (not done_0):
        if (stages_i[-1]==stages_i[0]) and (stages_i[-1]!=X.shape[1]) and (not done_0):
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stages_i, lengths, i)
            else:
                t_sample = sample_stage_var(model, stages_i, lengths, jumps_i, i)
            if obs_type == 'Fix':
                post = model.predict_proba(X_i, np.array([lengths[i]]))
            else:
                post = model.predict_proba(X_i, np.array([lengths[i]]), jumps_i)
            p_pred = np.dot(post,transmat)
            stage_pred = np.argmax(p_pred, axis=1)
            if not (stage_pred[-1]==stages_i[-1]):
                continue
            fig, ax = plt.subplots()
            time_disease = sojourns[stages_i.astype(int)]
            ax.errorbar(np.arange(nobs_i),stages_i,
                        yerr=np.std(t_sample, axis=0),
                        fmt='o',capsize=5,linestyle='solid',zorder=1,color=colors[0],linewidth=4)
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stage_pred, lengths, i)
            else:
                t_sample = sample_stage_var(model, stage_pred, lengths, jumps, i)
            ax.errorbar([nobs_i-1,nobs_i],[stages_i[-1],stage_pred[-1]],
                        yerr=[0, np.std(t_sample, axis=0)[-1]],
                        fmt='o',capsize=5,linestyle='solid',zorder=0,color=colors[0],alpha=0.5,linewidth=4)
            ax.set_title('Age at baseline = '+str(round(age[i],2)), fontsize=fsize)
            ax.set_xlabel('Observation time (years)', fontsize=fsize)
            ax.set_ylabel('Stage', fontsize=fsize)
            ax.set_yticks(np.arange(min([stages_i[-1],stage_pred[-1]]), max([stages_i[-1],stage_pred[-1]])+1, 1.0))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.tick_params(labelsize=fsize)
            plt.subplots_adjust(top=0.95, right=0.99, left=.12, bottom=.12)
            done_0 = 1
            continue
        #        if (stages_i[-1]-stages_i[0] == 1) and (stages_i[-1]==stages_i[-2]) and (not done_1):
        if (stages_i[-1]!=stages_i[0]) and (stages_i[-1]!=X.shape[1]) and (not done_1):
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stages_i, lengths, i)
            else:
                t_sample = sample_stage_var(model, stages_i, lengths, jumps_i, i)
            if obs_type == 'Fix':
                post = model.predict_proba(X_i, np.array([lengths[i]]))
            else:
                post = model.predict_proba(X_i, np.array([lengths[i]]), jumps_i)
            p_pred = np.dot(post,transmat)
            stage_pred = np.argmax(p_pred, axis=1)
            if not ((stage_pred[-1]-stages_i[-1])==1):
                continue
            fig, ax = plt.subplots()
            ax.errorbar(np.arange(nobs_i),stages_i,
                        yerr=np.std(t_sample, axis=0),
                        fmt='o',capsize=5,linestyle='dotted',zorder=1,color=colors[1],linewidth=4)
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stage_pred, lengths, i)
            else:
                t_sample = sample_stage_var(model, stage_pred, lengths, jumps_i, i)
            ax.errorbar([nobs_i-1,nobs_i],[stages_i[-1],stage_pred[-1]],
                        yerr=[0, np.std(t_sample, axis=0)[-1]],
                        fmt='o',capsize=5,linestyle='dotted',zorder=0,color=colors[1],alpha=0.5,linewidth=4)
            ax.set_title('Age at baseline = '+str(round(age[i],2)), fontsize=fsize)
            ax.set_xlabel('Observation time (years)', fontsize=fsize)
            ax.set_ylabel('Stage', fontsize=fsize)
            ax.set_yticks(np.arange(min([stages_i[-1],stage_pred[-1]])-2, max([stages_i[-1],stage_pred[-1]])+1, 1.0))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.tick_params(labelsize=fsize)
            plt.subplots_adjust(top=0.95, right=0.99, left=.12, bottom=.12)
            done_1 = 1
            continue
        #        if (stages_i[-1]-stages_i[0] == 2) and (not done_2):
        if (stages_i[-1]!=stages_i[0]) and (stages_i[-1]!=X.shape[1]) and (not done_2):
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stages_i, lengths, i)
            else:
                t_sample = sample_stage_var(model, stages_i, lengths, jumps_i, i)
            if obs_type == 'Fix':
                post = model.predict_proba(X_i, np.array([lengths[i]]))
            else:
                post = model.predict_proba(X_i, np.array([lengths[i]]), jumps_i)
            p_pred = np.dot(post,transmat)
            stage_pred = np.argmax(p_pred, axis=1)
            if not (stage_pred[-1]==stages_i[-1]):
                continue
            fig, ax = plt.subplots()
            ax.errorbar(np.arange(nobs_i),stages_i,
                        yerr=np.std(t_sample, axis=0),
                        fmt='o',capsize=5,linestyle='dashed',zorder=1,color=colors[2],linewidth=4)
            if obs_type=='Fix':
                t_sample = sample_stage_fix(model, stage_pred, lengths, i)
            else:
                t_sample = sample_stage_var(model, stage_pred, lengths, jumps_i, i)
            ax.errorbar([nobs_i-1,nobs_i],[stages_i[-1],stage_pred[-1]],
                        yerr=[0, np.std(t_sample, axis=0)[-1]],
                        fmt='o',capsize=5,linestyle='dashed',zorder=0,color=colors[2],alpha=0.5,linewidth=4)
            ax.set_title('Age at baseline = '+str(round(age[i],2)), fontsize=fsize)
            ax.set_xlabel('Observation time (years)', fontsize=fsize)
            ax.set_ylabel('Stage', fontsize=fsize)
            ax.set_yticks(np.arange(min([stages_i[-1],stage_pred[-1]])-3, max([stages_i[-1],stage_pred[-1]])+1, 1.0))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.tick_params(labelsize=fsize)
            plt.subplots_adjust(top=0.95, right=0.99, left=.1, bottom=.12)
            done_2 = 1
            continue

def crossval_new(dmmse, dx_change, X, lengths, jumps, labels, ids, transmat_prior_fw, n_components, obs_type):
    np.random.seed(0)
    n_iter = 100
    is_tebm = True
    is_gppm = False
    cut_nans = False
    baseline = False
    if baseline:
        n_iter = 0
        X_temp, lengths_temp, jumps_temp = [], [], []
        for i in range(len(lengths)):
            nobs_i = lengths[i]
            X_temp.append(X[np.sum(lengths[:i])])
            lengths_temp.append(1)
            jumps_temp.append(0)
        X = np.array(X_temp)
        lengths = np.array(lengths_temp)
        jumps = np.array(jumps_temp)

    biomarkers = ['Ventricles','Hippocampus','Entorhinal','Fusiform','MidTemp','WholeBrain','ABETA_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17','ADAS13','RAVLT_learning','MMSE','SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16']
    table = pd.read_csv('/home/paw/code/data_proc/out_data/adni_all_adjusted_units3months_icvcorrected_FORGPPM.csv')
    X_gppm, Y_gppm, ids_gppm, list_biomarker, group_gppm = GP_progression_model.convert_from_df(table, biomarkers)
    # get equivalent dataset for TEBM
    X_temp, lengths_temp, jumps_temp, labels_temp, dx_change_temp = [], [], [], [], []
    for i in range(len(ids)):
        nobs_i = lengths[i]
        X_i = X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        jumps_i = jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
        if ids[i] in ids_gppm:
            lengths_temp.append(nobs_i)
            labels_temp.append(labels[i])
            dx_change_temp.append(dx_change[i])
            for j in range(nobs_i):
                X_temp.append(X_i[j])
                jumps_temp.append(jumps_i[j])
    X = np.array(X_temp)
    jumps = np.array(jumps_temp)
    lengths = np.array(lengths_temp)
    labels = np.array(labels_temp)
    dx_change = np.array(dx_change_temp)

    ### cross-validation
    N_folds = 2
    print ('CV over',N_folds,'folds...')
    test_idxs, train_idxs = [], []
    cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
    # reshape data into 1 row / person for cv.split
    X_temp, jumps_temp = [], []
    for i in range(len(lengths)):
        nobs_i = lengths[i]
        X_temp.append(X[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
        jumps_temp.append(jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i])
    # shape: N_ppl, N_bms, N_tps
    X_temp = np.array(X_temp)
    jumps_temp = np.array(jumps_temp)
    
    """
    X_temp = np.array(X_temp)
    jumps_temp = np.array(jumps_temp)
    for train, test in cv.split(X_temp,np.array(labels)):
        test_idxs.append(test)
        train_idxs.append(train)
    test_idxs = np.array(test_idxs)
    train_idxs = np.array(train_idxs)
    """
    test_idxs, train_idxs = [], []
    for train, test in cv.split(np.array(X_gppm).T,np.array(group_gppm)):
        test_idxs.append(test)
        train_idxs.append(train)
    test_idxs = np.array(test_idxs)
    train_idxs = np.array(train_idxs)
   
    pred_all = []
    
    cv_auc_stage_bl, cv_auc_stage_delta, cv_auc_pred0, cv_auc_pred1 = [], [], [], []
    for idx in range(len(test_idxs)):
        lengths_train = lengths[train_idxs[idx]]
        labels_train = labels[train_idxs[idx]]
        # now reshape data back to long format: N_ppl*N_tps, N_bms
        X_train = X_temp[train_idxs[idx]]
        jumps_train = jumps_temp[train_idxs[idx]]
        """
        # for gppm
        X_train_gppm = X_train.copy()
        jumps_train_gppm = jumps_train.copy()
        tempX, templengths, tempT = [], [], []
        for i in range(len(X_train_gppm)):
            temptempX, temptempT = [], []
            flag = False
            for j in range(len(X_train_gppm[i])):
                if not np.isnan(X_train_gppm[i][j]).all():
                    temptempX.append(X_train_gppm[i][j])
                    temptempT.append(jumps_train_gppm[i][j])
                    flag = True
            tempX.append(np.array(temptempX))
            tempT.append(np.array(temptempT))
            if flag:
                templengths.append(len(temptempX))
        X_train_gppm = np.array(tempX)
        time_train_gppm = np.array(tempT)
        time_train_gppm = np.array([np.cumsum(x) for x in tempT])
        lengths_train_gppm = np.array(templengths)
        """
        # back to reshaping
        temp = []
        for j in range(len(X_train)):
            for k in range(len(X_train[j])):
                temp.append(X_train[j][k])
        X_train = np.array(temp)
        temp = []
        for j in range(len(jumps_train)):
            for k in range(len(jumps_train[j])):
                temp.append(jumps_train[j][k])
        jumps_train = np.array(temp)
        
        lengths_test = lengths[test_idxs[idx]]
        # now reshape data back to long format: N_ppl*N_tps, N_bms
        X_test = X_temp[test_idxs[idx]]
        temp = []
        for j in range(len(X_test)):
            for k in range(len(X_test[j])):
                temp.append(X_test[j][k])
        X_test = np.array(temp)

        jumps_test = jumps_temp[test_idxs[idx]]
        temp = []
        for j in range(len(jumps_test)):
            for k in range(len(jumps_test[j])):
                temp.append(jumps_test[j][k])
        jumps_test = np.array(temp)

        #FIXME: set dx_change externally from initial data read-in
        #        true_conv = return_labels(all_data_0, all_data_1, all_data_2)[test_idxs[idx]]
        true_conv = dx_change[test_idxs[idx]]
        
        if is_tebm:
            """
            lengths_gppm, jumps_gppm = [], []
            for i in range(len(ids)):
                nobs_i = lengths[i]
                jumps_i = jumps[np.sum(lengths[:i]):np.sum(lengths[:i])+nobs_i]
                if ids[i] in ids_gppm:
                    lengths_gppm.append(nobs_i)
                    for j in range(nobs_i):
                        jumps_gppm.append(jumps_i[j])
            lengths_gppm = np.array(lengths_gppm)
            jumps_gppm = np.array(jumps_gppm)
            """
            #            print ('EBM only')
            # first fit model on training data with forward-backward prior to get sequence
            model = tebm_var.MixtureTEBM(X=X_train, lengths=lengths_train, jumps=jumps_train,
                                         n_components=n_components, covariance_type="diag",
                                         time_mean = 1,
                                         n_iter=1, tol=1E-3,
                                         #                                         n_iter=0, tol=1E-3,
                                         init_params='s', params='st',
                                         verbose=False, allow_nan=True,
                                         fwd_only=True, order=1)
            ml_seq, mixtures = model._fit_tebm(labels_train, n_start=16, n_iter=100, n_cores=4, model_type='GMM', cut_controls=False)

            save_variables = {}
            save_variables["seq_model"] = ml_seq
            save_variables["mixtures"] = [x.theta for x in mixtures]
            save_variables["train_ids"] = np.array(ids_gppm)[train_idxs[idx]]
            save_variables["test_ids"] = np.array(ids_gppm)[test_idxs[idx]]
            pickle_file = open('./'+str(N_folds)+'FoldCrossVal/tebm_fold'+str(idx)+'.pickle', 'wb')
            #            pickle_file = open('./'+str(N_folds)+'FoldCrossVal/ebm_fold'+str(idx)+'.pickle', 'wb')
            # now refit with 100 its
            model = tebm_var.MixtureTEBM(X=X_train, lengths=lengths_train, jumps=jumps_train,
                                         n_components=n_components, covariance_type="diag",
                                         time_mean = 1,
                                         n_iter=100, tol=1E-3,
                                         #                                         n_iter=0, tol=1E-3,
                                         init_params='s', params='st',
                                         algorithm='viterbi', verbose=True, allow_nan=True,
                                         fwd_only=True, order=1)
            model.S = ml_seq[0]
            model.mixtures = mixtures
            model.prob_mat = get_prob_mat(X_train, mixtures)
            model.fit()

            transmat = sp.linalg.expm(np.real(model.Q_))
            stages_model, _ = model.predict(X_test, lengths_test, jumps_test)
            bl_stage_tebm = []
            for i in range(len(lengths_test)):
                nobs_i = lengths_test[i]
                bl_stage_tebm.append(stages_model[np.sum(lengths_test[:i])])
            bl_stage_tebm = np.array(bl_stage_tebm)
            save_variables["Q"] = model.Q_
            save_variables["X_train"] = X_train
            save_variables["lengths_train"] = lengths_train
            save_variables["jumps_train"] = jumps_train
            save_variables["X_test"] = X_test
            save_variables["lengths_test"] = lengths_test
            save_variables["jumps_test"] = jumps_test
            save_variables["stages_model_bl"] = bl_stage_tebm
            pickle_output = pickle.dump(save_variables, pickle_file)
            pickle_file.close()            
        else:
            Y_gppm_train = np.array(Y_gppm)[:, train_idxs[idx]]
            X_gppm_train = np.array(X_gppm)[:, train_idxs[idx]]
            group_gppm_train = np.array(group_gppm)[train_idxs[idx]]
            X_gppm_test = np.array(X_gppm)[:, test_idxs[idx]]
            Y_gppm_test = np.array(Y_gppm)[:, test_idxs[idx]]
            group_gppm_test = np.array(group_gppm)[test_idxs[idx]]
            """
            temp = []
            for i in range(len(time_gppm_train)):
                temptemp = []
                for j in range(X.shape[1]):
                    temptemp.append(time_gppm_train[i].T)
                temp.append(np.array(temptemp).T)
            time_gppm_train = temp#np.array(temp).reshape(X_gppm_train.shape[0], X.shape[1])
            temp = []
            for i in range(len(X_gppm_train)):                
                temp.append(X_gppm_train[i])
            X_gppm_train = temp#np.array(temp).reshape(X_gppm_train.shape[0], X.shape[1])
            print (X_gppm_train[0].shape)
            print (time_gppm_train[0].shape)
            #            time_gppm_train = time_gppm_train.T
            #            X_gppm_train = X_gppm_train.T
            """
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dict_monotonicity = {'Ventricles':1,
                                 'Hippocampus':-1,
                                 'Entorhinal':-1,
                                 'Fusiform':-1,
                                 'MidTemp':-1,
                                 'WholeBrain':-1,
                                 'ABETA_UPENNBIOMK9_04_19_17':-1,
                                 'TAU_UPENNBIOMK9_04_19_17':1,
                                 'PTAU_UPENNBIOMK9_04_19_17':1,
                                 'ADAS13':1,
                                 'RAVLT_learning':-1,
                                 'MMSE':-1,
                                 'SUMMARYSUVR_COMPOSITE_REFNORM_UCBERKELEYAV45_10_17_16':1}
            model = GP_progression_model.GP_Progression_Model(X_gppm_train,
                                                              Y_gppm_train,
                                                              names_biomarkers=biomarkers,
                                                              monotonicity=[dict_monotonicity[k] for k in dict_monotonicity.keys()],
                                                              trade_off=10,
                                                              groups=group_gppm_train,
                                                              group_names=['CN','MCI','AD'],
                                                              device=device)
            model.model = model.model.to(device)
            model.Optimize(N_outer_iterations=6, N_iterations=200, verbose=True, plot=False)
            threshold = []
            for i in range(len(biomarkers)):
                temp = []
                for x in Y_gppm_train[i][group_gppm_train==1]:
                    temp.append(x[0])
                threshold.append(np.mean(temp))
            time_thresholds = model.Threshold_to_time(threshold = threshold, save_fig = './', from_EBM = False)
            save_variables = {}
            save_variables["time_thresholds"] = time_thresholds
            save_variables["test_ids"] = np.array(ids_gppm)[test_idxs[idx]]
            pickle_file = open('./'+str(N_folds)+'FoldCrossVal/gppm_fold'+str(idx)+'.pickle', 'wb')
            pickle_output = pickle.dump(save_variables, pickle_file)
            pickle_file.close()
            model.Save(path='./'+str(N_folds)+'FoldCrossVal/gppm_fold'+str(idx)+'/')
            predictions_train = model.Predict(X_gppm_train, Y_gppm_train)
            optim_time_train = model.Diagnostic_predictions(predictions_train)
            predictions_test = model.Predict(X_gppm_test, Y_gppm_test)
            optim_time_test = model.Diagnostic_predictions(predictions_test)
            model.Save_predictions(predictions_train, './'+str(N_folds)+'FoldCrossVal/gppm_fold'+str(idx)+'_predictions_train.pickle')
            model.Save_predictions(predictions_test, './'+str(N_folds)+'FoldCrossVal/gppm_fold'+str(idx)+'_predictions_test.pickle')
            """
            max_like = -np.inf
            max_n = 0
            like_arr = []
            for n in range(1, 20):
                print ('n_states',n)
                model = cthmm_fix.GaussianCTHMM(X=X_train, lengths=lengths_train, 
                                                n_components=n, covariance_type="diag",
                                                n_iter=100, tol=1E-3,
                                                init_params='mcst', params='mcst',
                                                verbose=False, allow_nan=False)
                model.fit()
                like = model.score(X_train, lengths_train)
                like_arr.append(like)
                if like > max_like:
                    max_like = like
                    max_n = n
                    max_model = model
                else:
                    break
            print ('ML number of states',max_n,max_like)
            n_components = max_n
            # now refit with forward-only prior
            transmat_prior_fw = np.ones((n_components,n_components))
            for i in range(len(transmat_prior_fw)):
                transmat_prior_fw[i,:i] = 0.
                transmat_prior_fw[i] /= np.sum(transmat_prior_fw[i])            
            model = cthmm_fix.GaussianCTHMM(X=X_train, lengths=lengths_train, 
                                            n_components=n_components, transmat_prior=transmat_prior_fw, covariance_type="diag",
                                            n_iter=100, tol=1E-3,
                                            init_params='mcst', params='mcst',
                                            verbose=False, allow_nan=False)
            """
        #        model.fit()
        if is_gppm:
            bl_stage_tebm_train = []
            for i in range(len(optim_time_train)):
                bl_stage_tebm_train.append(optim_time_train[i])
            bl_stage_tebm_train = np.array(bl_stage_tebm_train).flatten()
            bl_stage_tebm_test = []
            for i in range(len(optim_time_test)):
                bl_stage_tebm_test.append(optim_time_test[i])
            bl_stage_tebm_test = np.array(bl_stage_tebm_test).flatten()
            #use AD group for threshold
            gppm_cut = np.percentile(bl_stage_tebm_train[group_gppm_train==1], 15)
            print ('gppm_cut',gppm_cut)
            pred_conv = (bl_stage_tebm_test>gppm_cut).astype(int)
        else:
            if obs_type=='Fix':
                transmat = model.transmat_
                stages_model, _ = model.predict(X_test, lengths_test)
            else:
                transmat = sp.linalg.expm(np.real(model.Q_))
                stages_model, _ = model.predict(X_test, lengths_test, jumps_test)
            ### done up to here
            bl_stage_tebm, delta_stage_tebm = [], []
            for i in range(len(lengths_test)):
                nobs_i = lengths_test[i]
                bl_stage_tebm.append(stages_model[np.sum(lengths_test[:i])])
                delta_stage_tebm.append(stages_model[np.sum(lengths_test[:i])+nobs_i-1]-stages_model[np.sum(lengths_test[:i])])
            bl_stage_tebm = np.array(bl_stage_tebm)
            delta_stage_tebm = np.array(delta_stage_tebm)
        #        true_fast = return_true_conv(dmmse[test_idxs[idx]], lengths_test, time_idx=1, predict_mri=False)
        #        true_fast = return_true_conv(X_test, lengths_test, time_idx=2, predict_mri=True)
        if not is_gppm:
            #            pred0_tebm, pred1_tebm = tebm_preds(model, X_test, lengths_test, jumps_test, bl_stage_tebm, time_idx=1, obs_type=obs_type)
            pred0_tebm, pred1_tebm = tebm_preds(model, X_test, lengths_test, jumps_test, bl_stage_tebm, 1, obs_type, 8)
            auc_pred0_tebm = roc_auc_score(true_conv, pred0_tebm)
            pred0_stage_tebm = np.array([x*y for x,y in zip(pred0_tebm,bl_stage_tebm)])
            pred0_stage_tebm /= np.max(pred0_stage_tebm)
            auc_pred1_tebm = roc_auc_score(true_conv, pred0_stage_tebm)
            auc_stage_tebm_bl, _ = calc_roc_mba(bl_stage_tebm, true_conv, n_components, is_tebm=True)
        else:
            auc_stage_tebm_bl = np.sum([1 if x==y else 0 for x,y in zip(pred_conv,true_conv)])/len(pred_conv)
        print ('auc_stage_tebm_bl',auc_stage_tebm_bl)
        cv_auc_stage_bl.append(auc_stage_tebm_bl)

        #        auc_stage_tebm_delta, _ = calc_roc_mba(delta_stage_tebm, true_conv, n_components, is_tebm=True)
        #        print ('auc_stage_tebm_delta',auc_stage_tebm_delta)
        #        cv_auc_stage_delta.append(auc_stage_tebm_delta)
        
        if not is_gppm:
            print ('auc_pred0_tebm',auc_pred0_tebm)
            cv_auc_pred0.append(auc_pred0_tebm)
            print ('auc_pred1_tebm',auc_pred1_tebm)
            cv_auc_pred1.append(auc_pred1_tebm)
        
    print (N_folds,'fold CV auc_stage_bl',np.mean(cv_auc_stage_bl),'std',np.std(cv_auc_stage_bl))
    print (N_folds,'fold CV auc_stage_delta',np.mean(cv_auc_stage_delta),'std',np.std(cv_auc_stage_delta))
    if not is_gppm:
        print (N_folds,'fold CV auc_pred0',np.mean(cv_auc_pred0),'std',np.std(cv_auc_pred0))
        print (N_folds,'fold CV auc_pred1',np.mean(cv_auc_pred1),'std',np.std(cv_auc_pred1))

"""
def plot_model(samples_sequence,
               samples_f,
               bm_labels,
               stage_zscore,
               stage_biomarker_index):
    colour_mat = np.array([[0,0,1],[1,0,1],[1,0,0]])
    temp_mean_f = np.mean(samples_f, 1)
    vals = np.sort(temp_mean_f)[::-1]
    vals = np.array([np.round(x * 100.) for x in vals]) / 100.
    ix = np.argsort(temp_mean_f)[::-1]
    N_S = samples_sequence.shape[0]
    N_bio = len(bm_labels)
    if N_S > 1:
        fig, ax = plt.subplots(1, N_S)
    else:
        fig, ax = plt.subplots()
    for i in range(N_S):
        this_samples_sequence = np.squeeze(samples_sequence[ix[i], :, :]).T
        markers = np.unique(stage_biomarker_index)
        N = this_samples_sequence.shape[1]
        confus_matrix = np.zeros((N, N))
        for j in range(N):
            confus_matrix[j, :] = sum(this_samples_sequence == j)
        confus_matrix /= float(max(this_samples_sequence.shape))
        zvalues = np.unique(stage_zscore)
        N_z                     = len(zvalues)
        confus_matrix_z = np.zeros((N_bio, N, N_z))
        for z in range(N_z):
            confus_matrix_z[stage_biomarker_index[stage_zscore == zvalues[z]], :, z] = confus_matrix[
                                                                                       (stage_zscore == zvalues[z])[0],
                                                                                       :]
        confus_matrix_c = np.ones((N_bio, N, 3))
        for z in range(N_z):
            this_confus_matrix = confus_matrix_z[:, :, z]
            this_colour = colour_mat[z, :]
            alter_level = this_colour == 0
            this_colour_matrix = np.zeros((N_bio, N, 3))
            this_colour_matrix[:, :, alter_level] = np.tile(this_confus_matrix[markers, :].reshape(N_bio, N, 1),
                                                            (1, 1, sum(alter_level)))
            confus_matrix_c = confus_matrix_c - this_colour_matrix
        # must be a smarter way of doing this, but subplots(1,1) doesn't produce an array...
        if N_S > 1:
            ax[i].imshow(confus_matrix_c, interpolation='nearest', cmap=plt.cm.Blues)
            ax[i].set_xticks(np.arange(N))
            ax[i].set_xticklabels(range(1, N+1), rotation=45) #, fontsize=15)
            ax[i].set_yticks(np.arange(N_bio))
            ax[i].set_yticklabels(np.array(bm_labels, dtype='object'), rotation=30, ha='right', rotation_mode='anchor') #, fontsize=15)
            for tick in ax[i].yaxis.get_major_ticks():
                tick.label.set_color('black')
            ax[i].set_ylabel('Biomarker name') #, fontsize=20)
            ax[i].set_xlabel('Event position') #, fontsize=20)
            ax[i].set_title('Group ' + str(i) + ' f=' + str(vals[i]))
        else:
            ax.imshow(confus_matrix_c, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_xticks(np.arange(N))
            ax.set_xticklabels(range(1, N+1), rotation=45) #, fontsize=15)
            ax.set_yticks(np.arange(N_bio))
            ax.set_yticklabels(np.array(bm_labels, dtype='object'), rotation=30, ha='right', rotation_mode='anchor') #, fontsize=15)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_color('black')
            ax.set_ylabel('Biomarker name') #, fontsize=20)
            ax.set_xlabel('Event position') #, fontsize=20)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_color('black')
        plt.tight_layout()
    return fig, ax
"""
