# Reading JMA radar data in netcdf format
# for time series forecast
import netCDF4
import numpy as np
import h5py

import glob
import subprocess
import sys
import os.path

# extract data from nc file
def ext_nc_JMA(fname):
    #nc = netCDF4.Dataset('../data/2015/01/01/2p-jmaradar5_2015-01-01_0000utc.nc', 'r')
    nc = netCDF4.Dataset(fname, 'r')
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
    #
    # extract data
    Rclip = R.data[0,j0:j1,i0:i1]
    #Rclip = Rclip.T   # transpose so that i index comes first
    lon_clip=lons.data[i0:i1]
    lat_clip=lats.data[j0:j1]
    print("rainfall min:max",Rclip.min(),Rclip.max())
    # take severl statistics
    rmax_200 = np.max(Rclip)
    rmean_200 = np.mean(Rclip)
    rmax_100 = np.max(Rclip[50:151,50:151]) # 100x100 max
    rmean_100 = np.mean(Rclip[50:151,50:151])
    rcent = Rclip[100,100]
    # save data
    return(rmax_200,rmean_200,rmax_100,rmean_100,rcent)

# read
#year =  '2015'
#year =  '2016'
year =  '2017'
infile_root = '../data/jma_radar/%s/' % year
print('input dir:',infile_root)

# outfile
outfile_root = '../data/data_kanto_ts/'
print('output dir:',infile_root)

flog = open(outfile_root+'/_max_log_%s.csv' % year,'w')
flog.write('time,rmax_200,rmean_200,rmax_100,rmean_100,rcent\n')

nt = 12

#for infile in sorted(glob.iglob(infile_root + '*00utc.nc.gz')):
for infile in sorted(glob.iglob(infile_root + '/*/*/*00utc.nc.gz')):
#for infile in sorted(glob.iglob(infile_root + '/*/*/*/*00utc.nc.gz')):
    # read 1hour data at a time
    # initialize with -999.0
    Rmax_200 = np.full((nt),-999.0,dtype=np.float32)
    Rmean_200 = np.full((nt),-999.0,dtype=np.float32)
    Rmax_100 = np.full((nt),-999.0,dtype=np.float32)
    Rmean_100 = np.full((nt),-999.0,dtype=np.float32)
    Rcent = np.full((nt),-999.0,dtype=np.float32)
    for i in range(12):
        shour = '{0:02d}utc'.format(i*5)
        in_zfile = infile.replace('00utc',shour)
        print('reading zipped file:',in_zfile)
        # '-k' option for avoiding removing gz file
        subprocess.run('gunzip -kf '+in_zfile,shell=True)
        in_nc=in_zfile.replace('.gz','')
        print('reading nc file:',in_nc)
        if os.path.exists(in_nc):
            rmax_200,rmean_200,rmax_100,rmean_100,rcent = ext_nc_JMA(in_nc)
            flog.write('%s,%f,%f,%f,%f,%f\n'
                       % (in_nc,rmax_200,rmean_200,rmax_100,rmean_100,rcent))
        else:
            print('nc file not found!!!',in_nc)
            flog.write('%s,-999,-999,-999,-999,-999\n' % in_nc)
            next
        Rmax_200[i] = rmax_200
        Rmean_200[i] = rmean_200
        Rmax_100[i] = rmax_100
        Rmean_100[i] = rmean_100
        Rcent[i] = rcent
        subprocess.run('rm '+in_nc,shell=True)

    # write to h5 file
    h5fname = infile.split('/')[-1]
    h5fname = h5fname.replace('.nc.gz','_ts.h5')
    print('writing h5 file:',h5fname)
    h5file = h5py.File(outfile_root+h5fname,'w')
    h5file.create_dataset('rmax_200',data= Rmax_200)
    h5file.create_dataset('rmean_200',data= Rmean_200)
    h5file.create_dataset('rmax_100',data= Rmax_100)
    h5file.create_dataset('rmean_100',data= Rmean_100)
    h5file.create_dataset('rcent',data= Rcent)
    h5file.close()
    #import pdb; pdb.set_trace()
    #sys.exit()
