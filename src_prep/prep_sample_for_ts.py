# Generate heavy-rain Sampled dataset for time-series prediction
import numpy as np
import h5py
import pandas as pd

import glob
import gzip
import subprocess
import sys
import os.path

def sample_flat_from_data(df,nmax=100):
    # flat sampling for given dataframe
    print("data length of original csv",len(df))
    # replace -999 (NA) with "0.0" * note that this will work only if NA occurs very rarely
    df["rmax_200"]=df["rmax_200"].clip_lower(0)
    df["rmean_200"]=df["rmean_200"].clip_lower(0)
    df["rmax_100"]=df["rmax_100"].clip_lower(0)
    df["rmean_100"]=df["rmean_100"].clip_lower(0)
    df["rcent"]=df["rcent"].clip_lower(0)

    # remove the beginning and the end 1h of the sequence to avoid shortage
    df_cut = df.iloc[12:-12].copy()
    #df_cut = df
    #import pdb;pdb.set_trace()

    # calculate histogram for area max
    c = pd.cut(df_cut["rmax_100"],
               np.append(-np.inf,np.arange(0,220,10)),
               labels=np.arange(-1,21))
    df_cut["category"]=c
    print(df_cut["category"].count())
    counts = pd.crosstab(index=df_cut["category"],columns="count")
    print("counts for each category")
    print(counts)
       
    # do the sampling 
    indices = counts.index.categories
    # initialize
    df_sam = pd.DataFrame()
    for ind in indices:
        print("index:",ind)
        df_tmp = df_cut.loc[df_cut["category"] == ind]
        if ind == 20:
            nsample = df_tmp.shape[0]
        else:
            nsample = min(df_tmp.shape[0],nmax)
        print("Number of samples: take ",nsample," among ",df_tmp.shape[0]," values")
        df_sam = pd.concat([df_sam,df_tmp.sample(n=nsample,random_state=0)])
    return(df,df_sam)

if __name__ == '__main__':

    df1 = pd.read_csv('../data/data_kanto_ts/_max_log_2015.csv')
    df2 = pd.read_csv('../data/data_kanto_ts/_max_log_2016.csv')
    df3 = pd.read_csv('../data/data_kanto_ts/_max_log_2017.csv')

    # training dataset
    df_tr = pd.concat([df1,df2])
    # validation dataset
    df_va = df3

    df_data_tr,df_sam_tr = sample_flat_from_data(df_tr,nmax=2000)
    df_data_va,df_sam_va = sample_flat_from_data(df_va,nmax=1000)

    # df_sam = df_sam.reset_index(drop=True)
    # save to file
    df_data_tr.to_csv('../data/data_kanto_ts/_train_data_2015-2016.csv')
    df_data_va.to_csv('../data/data_kanto_ts/_valid_data_2017.csv')
    df_sam_tr.to_csv("../data/ts_kanto_train_flatsampled_JMARadar.csv")
    df_sam_va.to_csv("../data/ts_kanto_valid_flatsampled_JMARadar.csv")
    

