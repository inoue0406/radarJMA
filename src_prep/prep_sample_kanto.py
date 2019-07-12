# Generate heavy-rain Sampled dataset to Kanto area
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

import glob
import gzip
import subprocess
import sys
import os.path

if __name__ == '__main__':

    df = pd.read_csv('../data/train_simple_JMARadar.csv')

    df['vmax'] = 0.0

    for n,row in df.iterrows():
        #print(n,row)
        # take statistics for "future" data in order to give consistency with alljapan
        fn = '../data/data_h5/'+row["fnext"]
        #print('read:',fn)
        h5file = h5py.File(fn,'r')
        rain_X = h5file['R'][()]
        h5file.close()
        df['vmax'][n] = np.max(rain_X)
   

    # calculate histogram for the past data
    c = pd.cut(df["vmax"],
               np.append(-np.inf,np.arange(0,220,10)),
               labels=np.arange(-1,21))
    df["category"]=c
    print(df["category"].count())
    counts = pd.crosstab(index=df["category"],columns="count")
    print("counts for each category")
    print(counts)

    # do the sampling 
    indices = counts.index.categories
    # initialize
    df_sam = pd.DataFrame()
    for ind in indices[1:]: # exclude negative and zero rains
        print("index:",ind)
        df_tmp = df.loc[df["category"] == ind]
        if ind == 20:
            nsample = df_tmp.shape[0]
        else:
            nsample = min(df_tmp.shape[0],100)
        print("Number of samples: take ",nsample," among ",df_tmp.shape[0]," values")
        df_sam = pd.concat([df_sam,df_tmp.sample(n=nsample,random_state=0)])

    df_sam = df_sam.reset_index(drop=True)
    # save to file
    df_sam.to_csv("../data/train_kanto_flatsampled_JMARadar.csv")
    

