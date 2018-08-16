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

# params for the training
#batch_size = 100
num_epochs = 10
batch_size = 20
learning_rate = 0.001
tdim_use = 6

# test for training data

train_dataset = JMARadarDataset(root_dir='../data/data_h5/',
                                csv_file='../data/train_simple_JMARadar.csv',
                                tdim_use=tdim_use,
                                #train=True,
                                transform=None)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# test for reading one batch 
#obj = next(iter(train_loader))
#obj['past'] #[torch.FloatTensor of size 100x12x200x200]
#obj['future'] #[torch.FloatTensor of size 100x12x200x200]

# define ConvLSTM layear
#convlstm = ConvLSTM(input_channels=12, hidden_channels=[24,12], kernel_size=3, step=5, effective_step=[4]).cuda()
# ConvLSTM Encoder Predictor
convlstm = CLSTM_EP(input_channels=1, hidden_channels=12, kernel_size=3).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(convlstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),
        #    sample_batched['future'].size())
        input = Variable(sample_batched['past']).cuda()
        target = Variable(sample_batched['future']).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = convlstm(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch+1, num_epochs, i_batch+1, len(train_dataset)//batch_size, loss.data[0]))
        #pdb.set_trace()
        #sys.exit()

# save the model
torch.save(convlstm,"trained_CLSTM.model")
