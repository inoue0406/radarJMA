import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os

import pdb

from jma_pytorch_dataset import *

train_dataset = JMARadarDataset(root_dir='../data/data_h5/',
                                csv_file='../data/train_simple_JMARadar.csv',
                                #train=True,
                                transform=None)

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


#pdb.set_trace()

# test for reading one batch 
#obj = next(iter(train_loader))
#obj['past'] #[torch.FloatTensor of size 100x12x200x200]
#obj['future'] #[torch.FloatTensor of size 100x12x200x200]

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['past'].size(),
          sample_batched['future'].size())
