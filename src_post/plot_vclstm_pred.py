#
# Plot Predicted Rainfall Data
# Variational LSTM Version
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data

import pandas as pd
import h5py
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from jma_pytorch_dataset import *
from regularizer import *
from convolution_lstm_mod import *
from train_valid_epoch import *
from colormap_JMA import Colormap_JMA

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

def plot_png_whole(fname,pic_tg,pic_pred,df_sampled,irand):
    cm = Colormap_JMA()
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle("Precip prediction starting at: "+fname, fontsize=20)
    for nt in range(6):
        #for nt in range(1,12,2):
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
    interval = dfp_sampled['rcategory'].iloc[i]
    str_int = 'I{}-{}_'.format(abs(interval.left),interval.right)
    plt.savefig(pic_path+'comp_pred_'+str_int+fname+str(irand)+'.png')
    plt.close()

def plot_png_ind(fname,pic_tg,pic_pred,df_sampled,irand):
    cm = Colormap_JMA()
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
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        # save as png
        i = df_sampled.index[df_sampled['fname']==fname]
        i = int(i.values)
        interval = mod_str_interval(df_sampled['rcategory'].iloc[i])
        nt_str = 'rand%02d_dt%02d' % (irand,nt)
        plt.savefig(pic_path+'comp_pred_'+interval+fname+nt_str+'.png')
        plt.close()

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,reg,mode='png_whole',rand_try=1):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset
    valid_dataset = JMARadarDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # load the saved model
    model = torch.load(model_fname)
    #model = CLSTM_EP(input_channels=1, hidden_channels=12,
    #                    kernel_size=3).cuda()
    #model.load_state_dict(torch.load(model_fname))
    # evaluation mode
    model.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):

        fnames = sample_batched['fnames_future']
        # skip if no overlap
        if len(set(fnames) &  set(df_sampled['fname'].values)) == 0:
            print("skipped batch:",i_batch)
            continue

        for irand in range(rand_try):
            # random prediction usning vae
            sample_all = torch.cat([sample_batched['past'][:,:,0,:,:],
                                    torch.zeros(sample_batched['past'][:,:,0,:,:].shape)],
                                   dim=1)
            in_vae = Variable(reg.fwd(sample_all)).cuda()
            in_lstm = Variable(reg.fwd(sample_batched['past'])).cuda()
            target = Variable(reg.fwd(sample_batched['future'])).cuda()
        
            # Forward
            output, mu, logvar = model(in_vae,in_lstm)

            # Output only selected data in df_sampled
            for n,fname in enumerate(fnames):
                if (not (fname in df_sampled['fname'].values)):
                    print('skipped:',fname)
                    continue
                # convert to cpu
                pic = target[n,:,0,:,:].cpu()
                pic_tg = reg.inv(pic.data.numpy())
                pic = output[n,:,0,:,:].cpu()
                pic_pred = reg.inv(pic.data.numpy())
                # print
                print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
                # plot
                if mode == 'png_whole': # output as stationary image
                    plot_png_ind(fname,pic_tg,pic_pred,df_sampled,irand)
                if mode == 'png_ind': # output as invividual image
                    plot_png_ind(fname,pic_tg,pic_pred,df_sampled,irand)
                # free GPU memory (* This seems to be necessary if going without backprop)
            del in_vae,in_lstm,target,output
        
if __name__ == '__main__':
    # params
    batch_size = 10
    tdim_use = 12

    data_path = '../data/data_h5/'
    filelist = '../data/valid_simple_JMARadar.csv'
    model_fname = 'result_20190708_vclstm_modtst/trained_CLSTM.model'
    pic_path = 'result_20190708_vclstm_modtst/png/'

    scaling = 'linear'
    
    # prepare regularizer for data
    if scaling == 'linear':
        reg = LinearRegularizer()
    elif scaling == 'log':
        reg = LogRegularizer()

    # samples to be plotted
    sample_path = '../data/sampled_forplot_one_JMARadar.csv'

    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
    
    plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path,reg,mode='png_ind',rand_try=10)


