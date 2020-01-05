# Generate heavy-rain Sampled dataset for time-series prediction
import numpy as np
import h5py
import pandas as pd

import glob
import gzip
import subprocess
import sys
import os.path

if __name__ == '__main__':

    df = pd.read_csv('../data/data_kanto_ts/_max_log_2015.csv')

    print("data length of original csv",len(df))
    # remove -999 (NA) row
    df = df[df['rmax_100'] > -0.001]
    print("data length of NA-removed csv",len(df))

    # calculate histogram for area max
    c = pd.cut(df["rmax_100"],
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
    for ind in indices:
        print("index:",ind)
        df_tmp = df.loc[df["category"] == ind]
        if ind == 20:
            nsample = df_tmp.shape[0]
        else:
            nsample = min(df_tmp.shape[0],1000)
        print("Number of samples: take ",nsample," among ",df_tmp.shape[0]," values")
        df_sam = pd.concat([df_sam,df_tmp.sample(n=nsample,random_state=0)])

    df_sam = df_sam.reset_index(drop=True)
    # save to file
    df_sam.to_csv("../data/ts_kanto_train_flatsampled_JMARadar.csv")
    

