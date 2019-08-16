#
# Plot Predicted Rainfall Data
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

from mnist_pytorch_dataset import *
from scaler import *
from convolution_lstm_mod import *
from train_valid_epoch import *
#from colormap_JMA import Colormap_JMA

# plot comparison of predicted vs ground truth
def plot_comp_prediction(model_fname,batch_size,tdim_use,
                         pic_path,scl,case,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset        
    valid_dataset = MNISTDataset(data_file='../data_mnist/mnist/mnist_valid_2000_data.hkl',
                                 source_file='../data_mnist/mnist/mnist_valid_2000_sources.hkl',
                                 tdim_use=tdim_use,
                                 transform=None)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # load the saved model
    convlstm = torch.load(model_fname)
    #convlstm = CLSTM_EP(input_channels=1, hidden_channels=12,
    #                    kernel_size=3).cuda()
    #convlstm.load_state_dict(torch.load(model_fname))
    # evaluation mode
    convlstm.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        if i_batch > 0:
            break
        fnames = sample_batched['fnames_future']
        # apply the trained model to the data
        input = Variable(scl.fwd(sample_batched['past'])).cuda()
        target = Variable(scl.fwd(sample_batched['future'])).cuda()
        #input = Variable(sample_batched['past']).cpu()
        #target = Variable(sample_batched['future']).cpu()
        output = convlstm(input)
        
        for n,fname in enumerate(fnames):
            # convert to cpu
            pic = target[n,:,0,:,:].cpu()
            pic_tg = scl.inv(pic.data.numpy())
            pic = output[n,:,0,:,:].cpu()
            pic_pred = scl.inv(pic.data.numpy())
            # print
            print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
            # plot
            # cm = Colormap_JMA()
            cm = 'Greys'
            if mode == 'png_whole': # output as stationary image
                fig, ax = plt.subplots(figsize=(20, 6))
                fig.suptitle("Precip prediction starting at: "+fname+"\n"+case, fontsize=20)
                for nt in range(10):
                #for nt in range(1,12,2):
                    id = nt
                    pos = nt+1
                    dtstr = str((id+1))
                    # target
                    plt.subplot(2,10,pos)
                    im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='upper')
                    plt.title("true:"+dtstr+"min")
                    plt.grid()
                    # predicted
                    plt.subplot(2,10,pos+10)
                    im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='upper')
                    plt.title("pred:"+dtstr+"steps")
                    plt.grid()
                fig.subplots_adjust(right=0.95)
                cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                # save as png
                plt.savefig(pic_path+'comp_pred_'+fname+str(i_batch)+str(n)+'.png')
                plt.close()
            if mode == 'png_ind': # output as invividual image
                for nt in range(10):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.suptitle("Precip prediction starting at: "+fname+"\n"+case, fontsize=10)
                    #        
                    id = nt
                    pos = nt+1
                    dtstr = str((id+1))
                    # target
                    plt.subplot(1,2,1)
                    im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='upper')
                    plt.title("true:"+dtstr+"steps")
                    plt.grid()
                    # predicted
                    plt.subplot(1,2,2)
                    im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='upper')
                    plt.title("pred:"+dtstr+"steps")
                    plt.grid()
                    # color bar
                    fig.subplots_adjust(right=0.93,top=0.85)
                    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    nt_str = '_dt%02d' % nt
                    plt.savefig(pic_path+'comp_pred_'+fname+str(i_batch)+str(n)+nt_str+'.png')
                    plt.close()
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output
        
if __name__ == '__main__':
    # params
    batch_size = 10
    tdim_use = 10

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 2:
        print('Usage: python plot_comp_prediction.py CASENAME')
        quit()

    case = argvs[1]
    #case = 'result_20190712_tr_clstm_flatsampled'
    #case = 'result_20190625_clstm_lrdecay07_ep20'

    model_fname = case + '/trained_CLSTM.model'
    pic_path = case + '/png/'

    data_scaling = 'linear'
    
    # prepare scaler for data
    if data_scaling == 'linear':
        scl = LinearScaler(rmax=255.0)

    plot_comp_prediction(model_fname,batch_size,tdim_use,
                         pic_path,scl,case,mode='png_ind')
    plot_comp_prediction(model_fname,batch_size,tdim_use,
                         pic_path,scl,case,mode='png_whole')


