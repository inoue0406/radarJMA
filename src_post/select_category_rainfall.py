#
# Select Plotting Candidates
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data

import pandas as pd
import h5py
import os
import sys

# -----------------------------
# add "src" as import path
path = os.path.join('/home/tsuyoshi/radarJMA/src')
sys.path.append(path)

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from train_valid_epoch import *

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
    # only select largest 2 category
    count = 0
    for cat in reversed(df.rcategory.cat.categories):
        df_slct = df[df['rcategory'] == cat]
        # random sample from data
        tmp_df = df_slct.sample(n=Nsample,random_state=0)
        df_sampled = df_sampled.append(tmp_df)
        count = count + 1
        if count >= 2:
            break
    df_sampled = df_sampled.reset_index()
    print('Selected Samples:')
    print(df_sampled)
    return(df_sampled)

# main routine
if __name__ == '__main__':
    # initialize random seed
    print('Select Plotting Candidates')
    np.random.seed(0)
    # params
    batch_size = 10
    tdim_use = 12

    data_path = '../data/data_h5/'
    filelist = '../data/valid_simple_JMARadar.csv'
    # samples to be plotted
    sample_path = '../data/sampled_forplot_JMARadar.csv'

    # sample and take data (This should take some time)
    df_sampled = select_category_rainfall(data_path,filelist,batch_size,tdim_use,
                                          Nsample=10)
    df_sampled.to_csv(sample_path)
    print('sampled list of paths written as:',sample_path)


