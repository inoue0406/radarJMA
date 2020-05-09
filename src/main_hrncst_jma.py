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

from jma_pytorch_dataset import *
from scaler import *
from test_persistence import *
from utils import Logger
from opts import parse_opts

# Forecat by

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
    
    # prepare scaler for data
    if opt.data_scaling == 'linear':
        scl = LinearScaler()
    elif opt.data_scaling == 'log':
        scl = LogScaler()
        
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

    # Test for Persistence Forecast
    loss_fn = torch.nn.MSELoss()
    # testing for the trained model
    for threshold in opt.eval_threshold:
        test_persistence(test_loader,loss_fn,
                         test_logger,opt,scl,threshold)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
