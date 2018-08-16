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
from valid_persistence_epoch import *
from utils import Logger
from opts import parse_opts

# Persistence Forecast

if __name__ == '__main__':
   
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
        
    # loading datasets (we only need validation data)
    valid_dataset = JMARadarDataset(root_dir=opt.data_path,
                                    csv_file=opt.train_path,
                                    tdim_use=opt.tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=False)
    
    # Prep logger
    valid_logger = Logger(
        os.path.join(opt.result_path, 'valid.log'),
        ['epoch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor'])

    # Validation for Persistence Forecast
    epoch = 0
    loss_fn = torch.nn.MSELoss()
    valid_persistence_epoch(epoch,opt.n_epochs,valid_loader,loss_fn,
                valid_logger,opt)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = (tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
