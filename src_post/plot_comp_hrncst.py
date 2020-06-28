import numpy as np
import pandas as pd
import h5py
import os
import sys
import time
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)
from colormap_JMA import Colormap_JMA

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


def plot_comp(file_list,tdim_use,result_path):
    
    for i,infile in enumerate(file_list):
        rain_pred,rain_true = get_model_true_pairs(infile)
        if rain_pred is None:
            print("skipped:")
            continue
        # output file
        fpng = infile.split("/")[-1]
        fpng = fpng.replace(".h5","")
        # rain_true and rain_pred must be in [time x height x width] dimension
        cm = Colormap_JMA()
        for nt in range(6):
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.suptitle("Precip prediction starting at: "+infile+"\n", fontsize=10)
            #
            id = nt
            dtstr = str((id+1)*5)
            # target
            plt.subplot(1,2,1)
            im = plt.imshow(rain_true[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
            plt.title("true:"+dtstr+"min")
            plt.grid()
            # predicted
            plt.subplot(1,2,2)
            im = plt.imshow(rain_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
            plt.title("pred:"+dtstr+"min")
            plt.grid()
            # color bar
            fig.subplots_adjust(right=0.93,top=0.85)
            cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            # save as png
            nt_str = '_dt%02d' % nt
            plt.savefig(result_path+'/png/comp_pred_'+fpng+nt_str+'.png')
            plt.close()
    return

if __name__ == '__main__':

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python main_hcncst_jma.py data_dir result_dir')
        quit()

    # samples to be plotted
    sample_path = '../data/sampled_forplot_3day_hrncst.csv'
    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
        
    # search directory
    infile_root = argvs[1]
    #infile_root = '../data/hrncst_kanto_rerun/'

    file_list = infile_root  + df_sampled["fname"].values
    file_list = file_list.tolist()

    tdim_use = 6
    result_path = argvs[2]
    #result_path = "result_20200510_hrncst"

    #print("list of files for png output:")
    #print(file_list)

    # create result dir
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    plot_comp(file_list,tdim_use,result_path)
                    
