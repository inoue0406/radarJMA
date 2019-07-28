#
# Plot Predicted Rainfall Data
# Using rainymotion optical-flow based method
#
import torch
import torchvision
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np
import pandas as pd
import h5py
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from jma_pytorch_dataset import *
from test_RMpred import *
from colormap_JMA import Colormap_JMA

# add rainymotion source dir
path = os.path.join('../../rainymotion')
sys.path.append(path)
from rainymotion.models import *
from rainymotion.utils import *

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,batch_size,tdim_use,
                         df_sampled,pic_path,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset
    valid_dataset = JMARadarDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    tdim_use=tdim_use,
                                    transform=None)
    #valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False)
    # we don't use ordinary dataloader
    # load only selected samples by indexing
    indices = df_sampled['index'].values
    sample_batched = data.dataloader.default_collate([valid_dataset[i] for i in indices])

    # apply the model to the data
    input = Variable(sample_batched['past'].float()).cpu()
    target = Variable(sample_batched['future'].float()).cpu()
    # prediction by optical flow
    output = target.clone()
    for n in range(input.data.shape[0]):
        # prediction by rainymotion
        nowcast_orig = RM_predictor(input.data[n,:,0,:,:].numpy(),tdim_use)
        output.data[n,:,0,:,:] = torch.from_numpy(nowcast_orig)

    fnames = sample_batched['fnames_future']
    # Output only selected data in df_sampled
    for n,fname in enumerate(fnames):
        if (not (fname in df_sampled['fname'].values)):
            print('sampled data and loaded files are inconsistent!!')
            print(fname)
            next
        # convert to cpu
        pic = target[n,:,0,:,:].cpu()
        pic_tg = pic.data.numpy()*201.0
        pic = output[n,:,0,:,:].cpu()
        pic_pred = pic.data.numpy()*201.0
        # print
        print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
        # plot
        cm = Colormap_JMA()
        if mode == 'png_whole': # output as stationary image
            fig, ax = plt.subplots(figsize=(20, 10))
            fig.suptitle("Precip prediction starting at: "+fname, fontsize=20)
            for nt in range(6):
                id = nt*2+1
                pos = nt+1
                dtstr = str((id+1)*5)
                # target
                plt.subplot(2,6,pos)
                im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                plt.title("true:"+dtstr+"min")
                plt.grid()
                # predicted
                plt.subplot(2,6,pos+6)
                im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                plt.title("pred:"+dtstr+"min")
                plt.grid()
            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            # save as png
            i = df_sampled.index[df_sampled['fname']==fname]
            i = int(i.values)
            interval = mod_str_interval(df_sampled['rcategory'].iloc[i])
            plt.savefig(pic_path+'comp_pred_'+interval+fname+'.png')
            plt.close()
        if mode == 'png_ind': # output as individual image
            for nt in range(6):
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.suptitle("Precip prediction starting at: "+fname, fontsize=10)
                #        
                id = nt*2+1
                pos = nt+1
                dtstr = str((id+1)*5)
                # target
                plt.subplot(1,2,1)
                im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                plt.title("true:"+dtstr+"min")
                plt.grid()
                # predicted
                plt.subplot(1,2,2)
                im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                plt.title("pred:"+dtstr+"min")
                plt.grid()
                # color bar
                fig.subplots_adjust(right=0.95)
                cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                # save as png
                i = df_sampled.index[df_sampled['fname']==fname]
                i = int(i.values)
                interval = mod_str_interval(df_sampled['rcategory'].iloc[i])
                nt_str = '_dt%02d' % nt
                plt.savefig(pic_path+'comp_pred_'+interval+fname+nt_str+'.png')
                plt.close()
    del input,target,output

if __name__ == '__main__':
    # params
    batch_size = 10
    tdim_use = 12

    data_path = '../data/data_h5/'
    filelist = '../data/valid_simple_JMARadar.csv'
    pic_path = 'result_20180827_oflow/png/'

    # samples to be plotted
    sample_path = '../data/sampled_forplot_JMARadar.csv'

    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
    
    plot_comp_prediction(data_path,filelist,batch_size,tdim_use,
                         df_sampled,pic_path,mode='png_ind')
