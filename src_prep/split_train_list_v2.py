#!/usr/bin/env python

# parse data directory to make train/test list

import glob
import pandas as pd
import numpy as np
import os

import pdb
import h5py
#pdb.set_trace()

dirpath = "../data/data_h5"
file_list = sorted(glob.glob(dirpath))

#setting = "train"
setting = "valid"

# initialize
max_list = []
min_list = []
mean_list = []
fnames = []
fnexts = []

if setting == "train":
    # select 2015 and 2016 for training
    dates = pd.date_range(start='2015-01-01 00:00', end='2016-12-31 22:00', freq='H')
elif setting == "valid":
    # select 2017 for validation
    dates = pd.date_range(start='2017-01-01 00:00', end='2017-12-31 22:00', freq='H')
    
# file format "2p-jmaradar5_2015-01-01_0000utc.h5"

# We choose loop through continuous times for missed-file checking and 
# checking for valid X-Y pairs

for n,date in enumerate(dates):
    print(date)
    fname = date.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    print(fname)
    fpath = os.path.join(dirpath,fname)
    
    # +1h data for Y
    date1 = date + pd.offsets.Hour()
    fname1 = date1.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    fpath1 = os.path.join(dirpath,fname1)

    # current data
    if os.path.exists(fpath):
        print('Exists:',fpath)
        fnames.append(fname)
        # read file and check the value
        h5file = h5py.File(fpath,'r')
        rain = h5file['R'].value
        h5file.close()
        # take max/min/mean of rainfall dataset
        # unit:[mm/h]
        max_list.append(rain.max())
        min_list.append(rain.min())
        mean_list.append(rain.mean())
    else:
        fnames.append(np.nan)
        max_list.append(np.nan)
        min_list.append(np.nan)
        mean_list.append(np.nan)
    # check for next hour
    if os.path.exists(fpath1):
        print('Exists Next Hour:',fpath1)
        fnexts.append(fname1)
    else:
        print('NOT Exist Next Hour:',fpath1)
        fnexts.append(np.nan)

df = pd.DataFrame({'fname':fnames,'fnext':fnexts,
                   'rmax':max_list,'rmin':min_list,'rmean':mean_list})

if setting == "train":
    df.to_csv('../data/summary_alldata_train_JMARadar.csv')
elif setting == "valid":
    df.to_csv('../data/summary_alldata_valid_JMARadar.csv')  
#pdb.set_trace()
        
