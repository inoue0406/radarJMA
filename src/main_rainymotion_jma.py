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
from test_RMpred import *
from utils import Logger
from opts import parse_opts

# Prediction by Optical Flow with RainyMotion
# and semi-Lagrangian advection

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
        
    # loading datasets (we only need testation data)
    test_dataset = JMARadarDataset(root_dir=opt.data_path,
                                    csv_file=opt.train_path,
                                    tdim_use=opt.tdim_use,
                                    transform=None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=False)
    # Prep logger
    test_logger = Logger(
        os.path.join(opt.result_path, 'test.log'),
        ['loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor'])

    # Test for Optical Flow Forecast
    loss_fn = torch.nn.MSELoss()
    test_RMpred(test_loader,loss_fn,
                test_logger,opt)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
