import numpy as np
import torch 
import torchvision
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
from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
if __name__ == '__main__':
   
    # parse command-line options
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

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')

    # prepare scaler for data
    if opt.data_scaling == 'linear':
        scl = LinearScaler()
    if opt.data_scaling == 'root':
        scl = RootScaler()
    if opt.data_scaling == 'root_int':
        scl = RootIntScaler()
    elif opt.data_scaling == 'log':
        scl = LogScaler()
        
    if not opt.no_train:
        # loading datasets
        train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                        csv_file=opt.train_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        valid_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                        csv_file=opt.valid_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=False)

        if opt.model_name == 'clstm':
            # convolutional lstm
            from models.convolution_lstm_mod import *
            convlstm = CLSTM_EP(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
        elif opt.model_name == 'clstm_skip':
            # convolutional lstm with skip connection
            from models.convolution_lstm_mod import *
            convlstm = CLSTM_EP3(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
        elif opt.model_name == 'clstm_multi':
            # convolutional lstm with multiple layers
            from models.convolution_lstm_multi import *
            convlstm = CLSTM_EP_MUL(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
        elif opt.model_name == 'clstm_2lyr':
            # convolutional lstm with 2 layers
            from models.convolution_lstm_2lyr import *
            convlstm = CLSTM_2lyr(input_channels=1, hidden_channels=opt.hidden_channels,
                                  kernel_size=opt.kernel_size,batch_size=opt.batch_size).cuda()
        elif opt.model_name == 'clstm_upper':
            # convolutional lstm upper layer only
            from models.convolution_lstm_2lyr import *
#            convlstm = CLSTM_upper(input_channels=1, hidden_channels=opt.hidden_channels,
#                                   kernel_size=opt.kernel_size,batch_size=opt.batch_size).cuda()
            convlstm = CLSTM_upper2(input_channels=1, hidden_channels=opt.hidden_channels,
                                  kernel_size=opt.kernel_size,batch_size=opt.batch_size).cuda()
    
        if opt.transfer_path != 'None':
            # Use pretrained weights for transfer learning
            print('loading pretrained model:',opt.transfer_path)
            convlstm = torch.load(opt.transfer_path)

        modelinfo.write('Model Structure \n')
        modelinfo.write(str(convlstm))
        count_parameters(convlstm,modelinfo)
        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif opt.loss_function == 'WeightedMSE':
            loss_fn = weighted_MSE_loss
        elif opt.loss_function == 'MaxMSE':
            loss_fn = max_MSE_loss(opt.loss_weights)
        elif opt.loss_function == 'MultiMSE':
            loss_fn = multi_MSE_loss(opt.loss_weights)

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(convlstm.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(convlstm.parameters(), lr=opt.learning_rate)
            
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
            
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss'])
    
        # training 
        for epoch in range(1,opt.n_epochs+1):
            # step scheduler
            scheduler.step()
            # training & validation
            train_epoch(epoch,opt.n_epochs,train_loader,convlstm,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt,scl)
            #valid_epoch(epoch,opt.n_epochs,valid_loader,convlstm,loss_fn,
            #            valid_logger,opt,scl)

            if epoch % opt.checkpoint == 0:
                # save the trained model for every checkpoint
                # (1) as binary 
                torch.save(convlstm,os.path.join(opt.result_path,
                                                 'trained_CLSTM_epoch%03d.model' % epoch))
                # (2) as state dictionary
                torch.save(convlstm.state_dict(),
                           os.path.join(opt.result_path,
                                        'trained_CLSTM_epoch%03d.dict' % epoch))
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
        test_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                        csv_file=opt.test_path,
                                        tdim_use=opt.tdim_use,
                                        transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=False)

        
        
        # testing for the trained model
        for threshold in opt.eval_threshold:
            test_CLSTM_EP(test_loader,convlstm,loss_fn,opt,scl,threshold)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
