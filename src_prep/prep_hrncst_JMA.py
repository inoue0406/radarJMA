#---------------------------------------------------
# Preprocess high-resolution nowcast data by JMA
#---------------------------------------------------
import netCDF4
import numpy as np
import h5py

import glob
import subprocess
import sys
import os.path

def grid_Kanto_jma_radar():
    '''
    This routine returns lon-lat grid, which is consistent with JMA radar data
    around Kanto Region in Japan
    '''
    nc = netCDF4.Dataset('../data/work/jma_radar/2p-jmaradar5_2017-12-31_2355utc.nc', 'r')
    #
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("JMA Radar data dims:",nx,ny,nt)
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
    Rclip = Rclip.T   # transpose so that i index comes first
    lon_clip=lons.data[i0:i1]
    lat_clip=lats.data[j0:j1]
    return lon_clip, lat_clip

def read_hrncst_smoothed(lons_kanto,lats_kanto):
    '''
    Read high resolution nowcast data and output in a smoothed grid

    '''
    # take min-max range with some margin
    lon_min = np.min(lons_kanto) - 0.1
    lon_max = np.max(lons_kanto) + 0.1
    lat_min = np.min(lats_kanto) - 0.1
    lat_max = np.max(lats_kanto) + 0.1

    nc = netCDF4.Dataset('../data/work/jma_hrncst/4p-hrncstprate_japan0250_2017-01-01_0000utc.nc', 'r')
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("dims:",nx,ny,nt)
    # extract variable
    lons = np.array(nc.variables['LON'][:])
    lats = np.array(nc.variables['LAT'][:])
    R = nc.variables['PRATE'][:] # numpy.ma.core.MaskedArray
    # long_name: precipitation rate
    # units: 1e-3 meter/hour -> [mm/h]
    # scale_factor: 0.01
    # add_offset: 0.0
    id_lons = (lons < lon_max) * (lons > lon_min)
    id_lats = (lats < lat_max) * (lats > lat_min)
    lons_rect = lons[id_lons]
    lats_rect = lats[id_lats]
    # "R[:,id_lats,id_lons]" does not seem to work..
    r_tmp =R[:,id_lats,:]
    r_rect =r_tmp[:,:,id_lons]
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    
    lons_kanto, lats_kanto = grid_Kanto_jma_radar()
    read_hrncst_smoothed(lons_kanto,lats_kanto)
    import pdb;pdb.set_trace()




