import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import pandas as pd
import h5py
import os
import re

def read_h5_and_prep(h5_name):
    if os.path.isfile(h5_name):
        h5file = h5py.File(h5_name,'r')
        R = h5file['R'][()]
        R = np.maximum(R,0) # replace negative value with 0
        R = R[:,None,:,:] # add "channel" dimension as 1
        h5file.close()
    else:
        print("h5 file NOT FOUND !!",h5_name)
        R = np.zeros([12,1,200,200])
    return R

def check_consistency(max_series,R):
    r1 = np.max(R[:,:,50:151,50:151],axis=(1,2,3))
    r2 = max_series.flatten()
    print("chk1 time sereies:",r1)
    print("chk2 spatial:",r2)
    return

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
        self.df_data = pd.read_csv(csv_data)
        print("data length of original csv",len(self.df_data))
        
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

class JMATSConvDataset(data.Dataset):
    def __init__(self,csv_data,csv_anno,use_var,root_dir,tdim_use=12,resize=200,transform=None):
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
        self.df_data = pd.read_csv(csv_data)
        print("data length of original csv",len(self.df_data))
        
        self.df_anno = pd.read_csv(csv_anno)
        print("number of selected samples",len(self.df_anno))
        
        self.resize = resize
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

        # extract necessary information
        str1_fname = re.search('(2p-jmaradar5.+utc)',df_past['time'].values[0]).group(1)
        str1_min = re.search('(..)utc',df_past['time'].values[0]).group(1)
        str1_sub = re.sub('..utc','00utc',str1_fname)

        str2_fname = re.search('(2p-jmaradar5.+utc)',df_past['time'].values[-1]).group(1)
        str2_min = re.search('(..)utc',df_past['time'].values[0]).group(1)
        str2_sub = re.sub('..utc','00utc',str2_fname)

        root_dir = '../data/data_kanto'
        if(str1_sub==str2_sub):
            # when the series starts at 00min, we only need 1 file
            h5_name = os.path.join(root_dir, str1_sub+'.h5')
            R = read_h5_and_prep(h5_name)
        else:
            # else, we need 2 files
            h5_name1 = os.path.join(root_dir, str1_sub+'.h5')
            R1 = read_h5_and_prep(h5_name1)
            h5_name2 = os.path.join(root_dir, str2_sub+'.h5')
            R2 = read_h5_and_prep(h5_name2)
            R = np.concatenate([R1,R2],axis=0)

        # select necessary elements
        id_start = int(int(str1_min)/5)
        R = R[id_start:(id_start+self.tdim_use),:,:,:]

        # resize image to suit convolutional structure
        # in the case of vgg16 this should be 128
        #                resnet this should be 224
        Rint = F.interpolate(torch.from_numpy(R),size=self.resize)

        # past series
        rain_past = df_past[['rmax_100']].to_numpy()
        # future series
        rain_future = df_future[['rmax_100']].to_numpy()

        # check 
        # check_consistency(rain_past,R)
        
        sample = {'features': Rint,
                  'past': rain_past,
                  'future': rain_future}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
