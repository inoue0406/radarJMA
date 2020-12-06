# Reading JMA radar data in netcdf format
# take statistics for the whole Japanese region
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import glob
import gzip
import subprocess
import sys
import os.path

import datetime

# extract data from nc file
def ext_nc_JMA_ij(fname,ii,jj):
    nc = netCDF4.Dataset(fname, 'r')
    eps = 0.001 # small number
    #
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("dims:",nx,ny,nt)
    #
    # ext variable
    lons = nc.variables['LON'][:]
    lats = nc.variables['LAT'][:]
    R = nc.variables['PRATE'][:]
    #
    # clip area around tokyo
    lat_tokyo = 35.681167
    lon_tokyo = 139.767052
    nx_clip = 200
    ny_clip = 200
    i0=np.argmin(np.abs(lons.data-lon_tokyo)) - int(nx_clip/2)
    j0=np.argmin(np.abs(lats.data-lat_tokyo)) - int(ny_clip/2)
    i1=i0+nx_clip
    j1=j0+ny_clip
    # choose ii and jj to span the whole dataset area
    Rslct = np.zeros(35)
    count = 0
    # offset
    ii = ii - 8
    jj = jj - 8
    #
    ii0 = i0 + nx_clip*ii
    jj0 = j0 + ny_clip*jj
    ii1 = ii0 + nx_clip
    jj1 = jj0 + ny_clip
    # extract data
    Rclip = R.data[0,jj0:jj1,ii0:ii1]
    Rclip = Rclip.T   # transpose so that i index comes first
    lon_clip=lons.data[i0:i1]
    lat_clip=lats.data[j0:j1]
    # save data
    return(Rclip)

def sample_alljapan_prep(year,df_sam):
    infile_root = '../data/jma_radar/%d/' % (year)
    print('dir:',infile_root)

    nx = 200
    ny = 200
    nt = 12
    
    for n,row in df_sam.iterrows():
        for infile in [row["data"],row["data_prev"]]:
            #infile = row["data"]
            ii = row["ii"]
            jj = row["jj"]
            # read 1hour data at a time
            # initialize with -999.0
            R1h = np.full((nt,nx,ny),-999.0,dtype=np.float32)
            for i in range(12):
                shour = '{0:02d}utc'.format(i*5)
                in_zfile = infile.replace('00utc',shour)
                print('reading zipped file:',in_zfile,ii,jj)
                # '-k' option for avoiding removing gz file
                subprocess.run('gunzip -kf '+in_zfile,shell=True)
                in_nc=in_zfile.replace('.gz','')
                print('reading nc file:',in_nc)
                if os.path.exists(in_nc):
                    Rclip = ext_nc_JMA_ij(in_nc,ii,jj)
                else:
                    print('nc file not found!!!',in_nc)
                    next
                R1h[i,:,:]=Rclip
                subprocess.run('rm '+in_nc,shell=True)
            # max during 1h for each grid
            h5fname = infile.split('/')[-1]
            h5fname = h5fname.replace('.nc.gz','_'+str(ii)+'_'+str(jj)+'.h5')
            # write to h5 file
            print('writing h5 file:',h5fname)
            h5file = h5py.File('../data/data_alljapan_'+str(year)+'/'+h5fname,'w')
            h5file.create_dataset('R',data= R1h)
            h5file.close()
    #sys.exit()

def fname_1h_ago(fname):
    f2 = fname.split("/")[-1]
    f2 = f2.replace("2p-jmaradar5_","").replace("utc.nc.gz","")
    dt = datetime.datetime.strptime(f2,'%Y-%m-%d_%H%M')
    
    # +1h data for Y
    date1 = dt - pd.offsets.Hour()
    fname1 = date1.strftime('../data/jma_radar/%Y/%m/%d/2p-jmaradar5_%Y-%m-%d_%H%Mutc.nc.gz')
    return fname1

if __name__ == '__main__':
    #for year in [2015,2016,2017]:
    #year = 2016
    year = 2010
    df = pd.read_csv("../data/stat_jma/stat_alljapan_"+str(year)+".txt")
    Nerr = np.sum(df["max_rain"]<=-32768)
    print("Number of irregular values",Nerr)

    c = pd.cut(df["max_rain"],
               np.append(-np.inf,np.arange(0,220,10)),
               labels=np.arange(-1,21))
    df["category"]=c
    print(df["category"].count())
    counts = pd.crosstab(index=df["category"],columns="count")
    print("counts for each category")
    print(counts)
    indices = counts.index.categories
    # initialize
    df_sam = pd.DataFrame()
    for ind in indices[1:]: # exclude negative and zero rains
        print("index:",ind)
        df_tmp = df.loc[df["category"] == ind]
        if ind == 20:
            nsample = df_tmp.shape[0]
        else:
            nsample = min(df_tmp.shape[0],1000)
        print("Number of samples: take ",nsample," among ",df_tmp.shape[0]," values")
        df_sam = pd.concat([df_sam,df_tmp.sample(n=nsample,random_state=0)])

    print("total number of sampled data:",df_sam.shape[0])
    # add previous dataset
    df_sam["data_prev"] = df_sam["data"]
    df_sam = df_sam.reset_index(drop=True)
    for n,row in df_sam.iterrows():
        df_sam["data_prev"].values[n] = fname_1h_ago(row["data"])
    # file list for training
    tr_list = df_sam[["data_prev","data","ii","jj"]]
    tr_list.columns = ["fname","fnext","ii","jj"]
    # h5 version of filename
    for n,row in tr_list.iterrows():
        ii = row["ii"]
        jj = row["jj"]
        tr_list["fname"].values[n] = row["fname"].split('/')[-1].replace('.nc.gz','_'+str(ii)+'_'+str(jj)+'.h5')
        tr_list["fnext"].values[n] = row["fnext"].split('/')[-1].replace('.nc.gz','_'+str(ii)+'_'+str(jj)+'.h5')
    # save to file
    tr_list.to_csv("../data/train_alljapan_"+str(year)+"_JMARadar.csv")
    # prepare sampled data
    sample_alljapan_prep(year,df_sam)
    #hist = np.histogram(df["max_rain"],range=(0,201))
    


    
