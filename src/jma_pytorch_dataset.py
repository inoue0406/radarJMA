import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os

# Pytorch custom dataset for JMA Radar data
# The class assumes the data to be in h5 format

class JMARadarDataset(data.Dataset):
    def __init__(self,csv_file,root_dir,tdim_use=12,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df_fnames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.transform = transform
        
    def __len__(self):
        return len(self.df_fnames)
        
    def __getitem__(self, index):
        # read X
        h5_name_X = os.path.join(self.root_dir, self.df_fnames.ix[index, 'fname'])
        h5file = h5py.File(h5_name_X,'r')
        rain_X = h5file['R'][()]
        rain_X = np.maximum(rain_X,0) # replace negative value with 0
        rain_X = rain_X[-self.tdim_use:,None,:,:] # add "channel" dimension as 1
        h5file.close()
        # read Y
        h5_name_Y = os.path.join(self.root_dir, self.df_fnames.ix[index, 'fnext'])
        h5file = h5py.File(h5_name_Y,'r')
        rain_Y = h5file['R'][()]
        rain_Y = np.maximum(rain_Y,0) # replace negative value with 0
        rain_Y = rain_Y[:self.tdim_use,None,:,:] # add "channel" dimension as 1
        h5file.close()
        # save
        fnames_past = self.df_fnames.ix[index, 'fname']
        fnames_future = self.df_fnames.ix[index, 'fnext']
        sample = {'past': rain_X, 'future': rain_Y,
                  'fnames_past':fnames_past,'fnames_future':fnames_future}

        if self.transform:
            sample = self.transform(sample)

        return sample
