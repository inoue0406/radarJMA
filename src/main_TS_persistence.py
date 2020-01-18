# Persistence Forecast to be used as a baseline
import numpy as np
import pandas as pd
import h5py
import os
import sys
import json
import time

import pdb

from jma_timeseries_dataset import *
from scaler import *
from test_ts_persistence import *
from utils import Logger
from opts_ts import parse_opts

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

    # test datasets if specified
    if opt.test:
        # prepare loader
        test_dataset = JMATSDataset(csv_data=opt.test_data_path,
                                     csv_anno=opt.test_anno_path,
                                     use_var=opt.use_var,
                                     root_dir=None,
                                     tdim_use=opt.tdim_use,
                                     transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=4,
                                                   drop_last=True,
                                                   shuffle=False)
        # testing for the trained model
        model = "persistence"
        test_epoch(test_loader,model,opt)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
