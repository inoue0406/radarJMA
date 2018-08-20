import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os
import sys
import json
import time

import pdb

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts

if __name__ == '__main__':
    # params for the training
    #batch_size = 100
    #num_epochs = 10
    #batch_size = 10
    #learning_rate = 0.01
    #tdim_use = 6
   
    # parse options
    opt = parse_opts()
    print(opt)
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()
        
    if not opt.no_train:
        # loading datasets
        train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                        csv_file=opt.train_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        valid_dataset = JMARadarDataset(root_dir=opt.data_path,
                                        csv_file=opt.valid_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=False)
    
        # ConvLSTM Encoder Predictor
        convlstm = CLSTM_EP(input_channels=1, hidden_channels=opt.hidden_channels,
                            kernel_size=opt.kernel_size).cuda()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(convlstm.parameters(), lr=opt.learning_rate)
    
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor'])
    
        # training 
        for epoch in range(opt.n_epochs):
            train_epoch(epoch,opt.n_epochs,train_loader,convlstm,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt)
            valid_epoch(epoch,opt.n_epochs,valid_loader,convlstm,loss_fn,
                        valid_logger,opt)

        # save the trained model
        # (1) as binary 
        torch.save(convlstm,os.path.join(opt.result_path, 'trained_CLSTM.model'))
        # (2) as state dictionary
        torch.save(convlstm.state_dict(),
                   os.path.join(opt.result_path, 'trained_CLSTM.dict'))

    # test datasets if specified
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_CLSTM.model')
            print('loading pretrained model:',model_fname)
            convlstm = torch.load(model_fname)
            loss_fn = torch.nn.MSELoss()
            
        # prepare loader
        test_dataset = JMARadarDataset(root_dir=opt.data_path,
                                        csv_file=opt.test_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=False)
        
        # testing for the trained model
        test_CLSTM_EP(test_loader,convlstm,loss_fn,opt)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
