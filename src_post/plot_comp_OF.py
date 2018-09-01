#
# Plot Predicted Rainfall Data
# Using Optical Flow
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
import matplotlib.animation as animation

# -----------------------------
# add "src" as import path
path = os.path.join('/home/tsuyoshi/radarJMA/src')
sys.path.append(path)

from jma_pytorch_dataset import *
from test_OFpred import *

def Colormap_JMA(n_bin=50):
    # custom colormap for rainfall
    # after JMA
    # https://www.jma.go.jp/jma/kishou/info/colorguide/120524_hpcolorguide.pdf
    # Colormap2
    cdict = {'red':   [(0,0.95,0.95),
                       (0.01,0.95,0.63),
                       (0.05,0.63,0.13),
                       (0.1,0.13,0),
                       (0.2,0,0.98),
                       (0.3,0.98,1),
                       (0.5,1,1),
                       (0.8,1,0.71),
                       (1,0.71,0.71)],
             'green': [(0,0.95,0.95),
                       (0.01,0.95,0.82),
                       (0.05,0.82,0.55),
                       (0.1,0.55,0.25),
                       (0.2,0.25,0.96),
                       (0.3,0.96,0.6),
                       (0.5,0.6,0.16),
                       (0.8,0.16,0),
                       (1,0,0)],
             'blue':  [(0,1,1),
                       (0.01,1,1),
                       (0.05,1,1),
                       (0.1,1,1),
                       (0.2,1,0),
                       (0.3,0,0),
                       (0.5,0,0),
                       (0.8,0,0.41),
                       (1,0.41,0.41)]}
    
    cmap_name="precip"
    cm = LinearSegmentedColormap(cmap_name, cdict, N=n_bin)
    return(cm)

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

    # apply the trained model to the data
    input = Variable(sample_batched['past']).cpu()
    target = Variable(sample_batched['future']).cpu()
    # prediction by optical flow
    output = target.clone()
    for n in range(input.data.shape[0]):
        output.data[n,:,0,:,:] = OF_predictor(input.data[n,:,0,:,:])
    #
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
            interval = df_sampled['rcategory'].iloc[i]
            interval = interval.replace('(','')
            interval = interval.replace(']','')
            interval = interval.replace(',','')
            interval = interval.replace(' ','_')
            plt.savefig(pic_path+'comp_pred_'+interval+fname+'.png')
            plt.close()
#                if mode == 'png_parts': # output as step-by step png
#                    fig = plt.figure()
#                    ims = []
#                    for i in range(12):
#                        print(i)
#                        im = plt.imshow(x_train[i,:,:],origin='lower',animated=True)
#                        ims.append([im])                  # グラフを配列 ims に追加
#                    ani = animation.ArtistAnimation(fig, ims, interval=100)
#                    plt.show() 
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
                         df_sampled,pic_path,mode='png_whole')
    #import pdb;pdb.set_trace()

