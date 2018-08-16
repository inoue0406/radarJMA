#!/usr/bin/env python

# prepare file list for training

import pandas as pd
import numpy as np
import os

#setting = "train"
setting = "valid"

if setting == "train":
    df = pd.read_csv('../data/summary_alldata_train_JMARadar.csv')
elif setting == "valid":
    df = pd.read_csv('../data/summary_alldata_valid_JMARadar.csv')

# remove if one of the filenames is empty
df_out = df.dropna(subset={'fname','fnext'})
df_out = df_out[['fname','fnext']]
df_out = df_out.reset_index(drop=True)

if setting == "train":
    df_out.to_csv('../data/train_simple_JMARadar.csv',index_label='n')
elif setting == "valid":
    df_out.to_csv('../data/valid_simple_JMARadar.csv',index_label='n')

