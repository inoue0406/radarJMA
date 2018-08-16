import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os
import sys

import pdb

from jma_pytorch_dataset import *
from convolution_lstm_mod import *

# test for training data

train_dataset = JMARadarDataset(root_dir='../data/data_h5/',
                                csv_file='../data/train_simple_JMARadar.csv',
                                #train=True,
                                transform=None)

#batch_size = 100
batch_size = 10

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

#pdb.set_trace()

# test for reading one batch 
#obj = next(iter(train_loader))
#obj['past'] #[torch.FloatTensor of size 100x12x200x200]
#obj['future'] #[torch.FloatTensor of size 100x12x200x200]

# define ConvLSTM layear
#convlstm = ConvLSTM(input_channels=12, hidden_channels=[24,12], kernel_size=3, step=5, effective_step=[4]).cuda()
# ConvLSTM Encoder Predictor
convlstm = CLSTM_EP(input_channels=1, hidden_channels=12, kernel_size=3).cuda()
loss_fn = torch.nn.MSELoss()

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['past'].size(),
          sample_batched['future'].size())
    input = Variable(sample_batched['past']).cuda()
    output = convlstm(input)
    pdb.set_trace()
    sys.exit()


