import numpy as np
import pandas as pd
import h5py
import os
import sys
import time
import glob

from test_persistence import *
from criteria_precip import *

# testing forecst accuracy by JMA hrncst

def get_model_true_pairs(infile):
    # prep prediction from '00utc' dataset
    print("reading 00utc:",infile)
    if not os.path.isfile(infile):
        print("file NOT found")
        return None,None
    h5file = h5py.File(infile,'r')
    rain_pred = h5file['R'][()]
    rain_pred = np.maximum(rain_pred,0) # replace negative value with
    rain_pred = rain_pred[1:,:,:] # remove initial
    
    # prep ground truth
    rain_true = np.zeros(rain_pred.shape)
    
    for i in range(1,7):
        # prep ground truth from dataset in each time step
        shour = '{0:02d}utc'.format(i*5)
        tmp_file = infile.replace('00utc',shour)
        print("reading:",tmp_file)
        if not os.path.isfile(tmp_file):
            print("file NOT found")
            return None,None
        h5file = h5py.File(tmp_file,'r')
        rain_tmp = h5file['R'][()]
        rain_tmp= np.maximum(rain_tmp,0) # replace negative value with 0
        
        rain_true[i-1,:,:] = rain_tmp[0,:,:] # use initial data as the ground truth
    return rain_pred,rain_true


def eval_on_threshold(file_list,threshold,tdim_use,result_path):
    # evaluate metrics for single threshold
    
    # initialize
    SumSE_all = np.empty((0,tdim_use),float)
    hit_all = np.empty((0,tdim_use),float)
    miss_all = np.empty((0,tdim_use),float)
    falarm_all = np.empty((0,tdim_use),float)
    m_xy_all = np.empty((0,tdim_use),float)
    m_xx_all = np.empty((0,tdim_use),float)
    m_yy_all = np.empty((0,tdim_use),float)
    MaxSE_all = np.empty((0,tdim_use),float)
    FSS_t_all = np.empty((0,tdim_use),float)
    flist_all = []

    for i,infile in enumerate(file_list):
        rain_pred,rain_true = get_model_true_pairs(infile)
        if rain_pred is None:
            print("skipped:")
            continue
        # input must be in [sample x time x channels x height x width] dimension
        rain_true = rain_true[None,:,None,:,:]
        rain_pred = rain_pred[None,:,None,:,:]
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(rain_true,rain_pred,th=threshold)
        FSS_t = FSS_for_tensor(rain_true,rain_pred,th=threshold,win=10)
        # stat
        SumSE_all = np.append(SumSE_all,SumSE,axis=0)
        hit_all = np.append(hit_all,hit,axis=0)
        miss_all = np.append(miss_all,miss,axis=0)
        falarm_all = np.append(falarm_all,falarm,axis=0)
        m_xy_all = np.append(m_xy_all,m_xy,axis=0)
        m_xx_all = np.append(m_xx_all,m_xx,axis=0)
        m_yy_all = np.append(m_yy_all,m_yy,axis=0)
        MaxSE_all = np.append(MaxSE_all,MaxSE,axis=0)
        FSS_t_all = np.append(FSS_t_all,FSS_t,axis=0)
        #
        flist_all.append(infile)
        #if i > 10:
        #    break

    # ------------------------------------------------------------------
    # time
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,
                                          MaxSE_all,FSS_t_all,axis=(0))
    
    # save evaluated metric as csv file
    tpred = (np.arange(tdim_use)+1.0)*5.0 # in minutes
    # import pdb; pdb.set_trace()
    df = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE,
                       'CSI':CSI,
                       'FAR':FAR,
                       'POD':POD,
                       'Cor':Cor,
                       'MaxMSE': MaxMSE,
                       'FSS_mean': FSS_mean,
                       })
    df.to_csv(os.path.join(result_path,
                           'test_evaluation_predtime_%.2f.csv' % threshold), float_format='%.3f')
    # ------------------------------------------------------------------
    # samples
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,
                                          MaxSE_all,FSS_t_all,axis=(1))
    
    # save evaluated metric as csv file
    df = pd.DataFrame({'file':flist_all,
                       'RMSE':RMSE,
                       'CSI':CSI,
                       'FAR':FAR,
                       'POD':POD,
                       'Cor':Cor,
                       'MaxMSE': MaxMSE,
                       'FSS_mean': FSS_mean,
                       })
    df.to_csv(os.path.join(result_path,
                           'test_evaluation_samples_%.2f.csv' % threshold), float_format='%.3f')
    return

if __name__ == '__main__':

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python main_hcncst_jma.py data_dir result_dir')
        quit()
        
    # search directory
    infile_root = argvs[1]
    #infile_root = '../data/hrncst_kanto_rerun/'
    file_list = sorted(glob.iglob(infile_root + '/*00utc.h5'))
    
    thresholds = [0.5,10,20]
    tdim_use = 6
    result_path = argvs[2]
    #result_path = "result_20200510_hrncst"

    # create result dir
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    for threshold in thresholds:
        print("evaluation for the threshold ",threshold)
        eval_on_threshold(file_list,threshold,tdim_use,result_path)
                    
