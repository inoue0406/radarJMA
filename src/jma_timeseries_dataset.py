import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os

# Pytorch custom dataset for JMA timeseries data

class JMATSDataset(data.Dataset):
    def __init__(self,csv_data,csv_anno,use_var,root_dir,tdim_use=12,transform=None):
        """
        Args:
            csv_data (string): Path to the csv file with time series data.
            csv_anno (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # read time series data
        #csv_data = "../data/data_kanto_ts/_max_log_2015.csv"
        self.df_data = pd.read_csv(csv_data)
        print("data length of original csv",len(self.df_data))
        
        #csv_anno = "../data/ts_kanto_train_flatsampled_JMARadar.csv"
        self.df_anno = pd.read_csv(csv_anno)
        print("number of selected samples",len(self.df_anno))
        
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.use_var = use_var
        self.transform = transform
        
    def __len__(self):
        return len(self.df_anno)
        
    def __getitem__(self, index):
        
        # get training sample based on time series data and annotation
        idx = self.df_anno.iloc[index,0]
        df_past = self.df_data.iloc[(idx-self.tdim_use+1):(idx+1)]
        df_future = self.df_data.iloc[(idx+1):(idx+self.tdim_use+1)]

        # Features (independent variables) to be used as "X"
        rain_features = df_past[self.use_var].to_numpy() # the resulting tensor has [use_var X tdim_use] dimension
        # past series
        rain_past = df_past[['rmax_100']].to_numpy()
        # future series
        rain_future = df_future[['rmax_100']].to_numpy()
        
        sample = {'features': rain_features,
                  'past': rain_past,
                  'future': rain_future}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
