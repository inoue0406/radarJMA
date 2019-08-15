import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import hickle as hkl
import os

# Pytorch custom dataset for Moving MNIST data
# The class assumes the data to be in hkl format

class MNISTDataset(data.Dataset):
    def __init__(self,data_file,source_file,tdim_use=10,transform=None):
        """
        Args:
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.tdim_use = tdim_use
        self.transform = transform

        # load data
        print('reading from data file and source file:',data_file,source_file)
        self.data = hkl.load(data_file)
        # dimension change (n,H,W,ch) to channels first (n,ch,H,W)
        self.data = np.transpose(self.data,(0,3,1,2))
        
        self.sources = hkl.load(source_file)
        # number of samples
        self.N = int(self.data.shape[0]/(tdim_use*2))
        print('Number of samples:',self.N)
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, index):
        istart = index*(self.tdim_use*2)
        data_X = self.data[istart:(istart+self.tdim_use),:,:,:]
        data_Y = self.data[(istart+self.tdim_use):(istart+self.tdim_use*2),:,:,:]
        sample = {'past': data_X, 'future': data_Y,
                  'fnames_past':'past','fnames_future':'future'}

        if self.transform:
            sample = self.transform(sample)

        return sample
