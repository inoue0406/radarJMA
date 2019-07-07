# Reading JMA radar data in netcdf format
# take statistics for the whole Japanese region
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import h5py

import glob
import gzip
import subprocess
import sys
import os.path

# extract data from nc file
def ext_nc_JMA(fname):
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
    for ii in range(-8,2+1):
        for jj in range(-8,7+1):
            ii0 = i0 + nx_clip*ii
            jj0 = j0 + ny_clip*jj
            ii1 = ii0 + nx_clip
            jj1 = jj0 + ny_clip
            # extract data
            Rclip = R.data[0,jj0:jj1,ii0:ii1]
            Rclip = Rclip.T   # transpose so that i index comes first
            lon_clip=lons.data[i0:i1]
            lat_clip=lats.data[j0:j1]
            # ratio of valid value
            rate_valid = 1- np.sum(Rclip < -eps)/Rclip.size
            if rate_valid > (1.0-eps):
                print(ii+8,jj+8,ii0,jj0,lons.data[ii0],lats.data[jj0],ii1,jj1,lons.data[ii1],lats.data[jj1],rate_valid,Rclip.min(),Rclip.max())
                Rslct[count] = Rclip.max()
                count = count + 1
                #print("rainfall min:max",Rclip.min(),Rclip.max())
    import pdb;pdb.set_trace()
    # save data
    return(Rslct)

# read
infile_root = '../data/jma_radar/2017/'
print('dir:',infile_root)

nx = 200
ny = 200
nt = 12

for infile in sorted(glob.iglob(infile_root + '/*/*/*00utc.nc.gz')):
    # read 1hour data at a time
    # initialize with -999.0
    R1h = np.full((nt,nx,ny),-999.0,dtype=np.float32)
    for i in range(12):
        shour = '{0:02d}utc'.format(i*5)
        in_zfile = infile.replace('00utc',shour)
        print('reading zipped file:',in_zfile)
        # '-k' option for avoiding removing gz file
        subprocess.run('gunzip -kf '+in_zfile,shell=True)
        in_nc=in_zfile.replace('.gz','')
        print('reading nc file:',in_nc)
        if os.path.exists(in_nc):
            Rclip = ext_nc_JMA(in_nc)
        else:
            print('nc file not found!!!',in_nc)
            next
        R1h[i,:,:]=Rclip
        subprocess.run('rm '+in_nc,shell=True)
    # write to h5 file
    h5fname = infile.split('/')[-1]
    h5fname = h5fname.replace('.nc.gz','.h5')
    print('writing h5 file:',h5fname)
    h5file = h5py.File('../data_h5/'+h5fname,'w')
    h5file.create_dataset('R',data= R1h)
    h5file.close()
    #sys.exit()
        

        

