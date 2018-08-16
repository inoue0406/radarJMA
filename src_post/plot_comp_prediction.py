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

def select_category_rainfall(data_path,filelist,batch_size,tdim_use,
                             Nsample = 10):
    # select data by rainfall intensity
    # dataset
    valid_dataset = JMARadarDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    
    # collect rainfall intentity for the whole validation dataset
    print('Scanning for the whole validation dataset:')
    N = len(valid_loader.dataset)
    rain_maxes = []
    fnames = []
    for i_batch, sample_batched in enumerate(valid_loader):
        target = Variable(sample_batched['future']).cpu().data.numpy()*201.0
        fname_batch = sample_batched['fnames_future']
        # calc max for every 1hour
        v = np.max(target,axis=(1,2,3,4))
        rain_maxes.extend(v.tolist())
        fnames.extend(fname_batch)
    #set up bins
    bin = [-0.01,0.1,0.5,1,5,10,50,250]
    #attribute the values into its specific bins
    category = pd.cut(rain_maxes,bin)
    print('No. of samples by rainfall intensity:')
    print(category.value_counts())
    #
    df = pd.DataFrame({'fname':fnames,
                       'rmax':rain_maxes,
                       'rcategory':pd.Categorical(category)})
    # sample small number of data for visualization
    df_sampled = pd.DataFrame(columns = ['fname', 'rmax', 'rcategory'])
    for cat in df.rcategory.cat.categories:
        df_slct = df[df['rcategory'] == cat]
        # random sample from data
        tmp_df = df_slct.sample(n=Nsample,random_state=0)
        df_sampled = df_sampled.append(tmp_df)
    df_sampled = df_sampled.reset_index()
    print('Selected Samples:')
    print(df_sampled)
    return(df_sampled)

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path):
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
    convlstm = torch.load(model_fname)
    #convlstm = CLSTM_EP(input_channels=1, hidden_channels=12,
    #                    kernel_size=3).cuda()
    #convlstm.load_state_dict(torch.load(model_fname))
    # evaluation mode
    convlstm.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        # apply the trained model to the data
        input = Variable(sample_batched['past']).cuda()
        target = Variable(sample_batched['future']).cuda()
        #input = Variable(sample_batched['past']).cpu()
        #target = Variable(sample_batched['future']).cpu()
        output = convlstm(input)
        #
        fnames = sample_batched['fnames_future']
        # Output only selected data in df_sampled
        for n,fname in enumerate(fnames):
            if (fname in df_sampled['fname'].values):
                # convert to cpu
                pic = target[n,:,0,:,:].cpu()
                pic_tg = pic.data.numpy()*201.0
                pic = output[n,:,0,:,:].cpu()
                pic_pred = pic.data.numpy()*201.0
                # print
                print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
                # plot
                cm = Colormap_JMA()
                fig, ax = plt.subplots(figsize=(10, 20))
                fig.suptitle("Precip prediction starting at: "+fname, fontsize=20)
                for nt in range(6):
                    pos = nt*2+1
                    dtstr = str((nt+1)*5)
                    # target
                    plt.subplot(6,2,pos)
                    plt.imshow(pic_tg[nt,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("true:"+dtstr+"min")
                    plt.grid()
                    # predicted
                    plt.subplot(6,2,pos+1)
                    plt.imshow(pic_pred[nt,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
                    plt.title("pred:"+dtstr+"min")
                    plt.colorbar()
                    plt.grid()
                # save as png
                i = df_sampled.index[df_sampled['fname']==fname]
                i = int(i.values)
                interval = df_sampled['rcategory'].iloc[i]
                str_int = 'I{}-{}_'.format(abs(interval.left),interval.right)
                plt.savefig(pic_path+'comp_pred_'+str_int+fname+'.png')
                plt.close()
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output

# -----------------------------
# add "src" as import path
path = os.path.join('/home/tsuyoshi/radarJMA/src')
sys.path.append(path)

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from train_valid_epoch import *

# params
batch_size = 10
tdim_use = 6

data_path = '../data/data_h5/'
filelist = '../data/valid_simple_JMARadar.csv'
model_fname = 'result_20180814_base/trained_CLSTM.model'
pic_path = 'result_20180814_base/png/'

df_sampled = select_category_rainfall(data_path,filelist,batch_size,tdim_use,
                                      Nsample=10)
plot_comp_prediction(data_path,filelist,model_fname,batch_size,tdim_use,
                         df_sampled,pic_path)
#import pdb;pdb.set_trace()

