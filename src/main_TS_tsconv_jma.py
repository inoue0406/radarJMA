# seq2seq LSTM (no-convolutional model) for time series prediction
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

from jma_timeseries_dataset import *
from scaler import *
from train_valid_epoch_tsconv import *
from utils import Logger
from opts_ts import parse_opts

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    print("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
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
        
    if not opt.no_train:
        # loading datasets
        train_dataset = JMATSConvDataset(csv_data=opt.train_data_path,
                                     csv_anno=opt.train_anno_path,
                                     use_var=opt.use_var,
                                     root_dir=None,
                                     tdim_use=opt.tdim_use,
                                     transform=None)
    
        valid_dataset = JMATSConvDataset(csv_data=opt.valid_data_path,
                                     csv_anno=opt.valid_anno_path,
                                     use_var=opt.use_var,
                                     root_dir=None,
                                     tdim_use=opt.tdim_use,
                                     transform=None)
        
        #tstdata = next(iter(train_dataset))
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=7,
                                                   drop_last=True,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=7,
                                                   drop_last=True,
                                                   shuffle=False)

        if opt.model_name == 'seq2seq':
            # lstm seq2seq model
            CONV_HID_DIM = 32
            INPUT_DIM = 1 + CONV_HID_DIM
            OUTPUT_DIM = 1
            HID_DIM = 512
            N_LAYERS = 3
            ENC_DROPOUT = 0.5
            DEC_DROPOUT = 0.5

            from models.seq2seq_convlstm_ts import *
            enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
            dec = Decoder(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
            model = Seq2SeqConv(enc, dec, CONV_HID_DIM, device='cuda').cuda()
    
        if opt.transfer_path != 'None':
            # Use pretrained weights for transfer learning
            print('loading pretrained model:',opt.transfer_path)
            model = torch.load(opt.transfer_path)

        modelinfo.write('Model Structure \n')
        modelinfo.write(str(model))
        count_parameters(model,modelinfo)
#        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate)

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
            
            if epoch < 10:
                # freeze conv_encoder for first 10 epochs
                submodel = next(iter(model.children()))
                for param in submodel.parameters():
                    param.requires_grad = False
            else:
                # unfreeze conv_encoder for the rest
                submodel = next(iter(model.children()))
                for param in submodel.parameters():
                    param.requires_grad = True
            count_parameters(model,modelinfo)
            #import pdb;pdb.set_trace()
            
            # step scheduler
                    
            scheduler.step()
            # training & validation
            train_epoch(epoch,opt.n_epochs,train_loader,model,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt,scl)
            valid_epoch(epoch,opt.n_epochs,valid_loader,model,loss_fn,
                        valid_logger,opt,scl)

            if epoch % opt.checkpoint == 0:
                # save the trained model for every checkpoint
                # (1) as binary 
                torch.save(model,os.path.join(opt.result_path,
                                                 'trained_seq2seq_epoch%03d.model' % epoch))
                # (2) as state dictionary
                torch.save(model.state_dict(),
                           os.path.join(opt.result_path,
                                        'trained_seq2seq_epoch%03d.dict' % epoch))
        # save the trained model
        # (1) as binary 
        torch.save(model,os.path.join(opt.result_path, 'trained_seq2seq.model'))
        # (2) as state dictionary
        torch.save(model.state_dict(),
                   os.path.join(opt.result_path, 'trained_seq2seq.dict'))

    # test datasets if specified
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_seq2seq.model')
            print('loading pretrained model:',model_fname)
            model = torch.load(model_fname)
            loss_fn = torch.nn.MSELoss()
            
        # prepare loader
        test_dataset = JMATSConvDataset(csv_data=opt.test_data_path,
                                     csv_anno=opt.test_anno_path,
                                     use_var=opt.use_var,
                                     root_dir=None,
                                     tdim_use=opt.tdim_use,
                                     transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                                   batch_size=opt.batch_size,
                                                   batch_size=3, # small batch size used
                                                   num_workers=7,
                                                   drop_last=True,
                                                   shuffle=False)
        
        # testing for the trained model
        test_epoch(test_loader,model,loss_fn,opt,scl)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
