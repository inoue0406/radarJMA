#---------------------------------------------------
# Preprocess high-resolution nowcast data by JMA
#---------------------------------------------------
import glob
import subprocess
import sys
import os.path

import netCDF4
import numpy as np
import h5py

import scipy.ndimage
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

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

def read_hrncst_smoothed(lons_kanto,lats_kanto,fname):
    '''
    Read high resolution nowcast data and output in a smoothed grid

    '''
    # take min-max range with some margin
    delta = 1.0
    lon_min = np.min(lons_kanto) - delta
    lon_max = np.max(lons_kanto) + delta
    lat_min = np.min(lats_kanto) - delta
    lat_max = np.max(lats_kanto) + delta

    #nc = netCDF4.Dataset('../data/work/jma_hrncst/4p-hrncstprate_japan0250_2017-08-01_1100utc.nc', 'r')
    nc = netCDF4.Dataset(fname, 'r')
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
    r_rect =np.array(r_tmp[:,:,id_lons])
    r_rect = np.maximum(r_rect,0) # replace negative value with 0
    
    tdim = r_rect.shape[0] # time dimension for nowcast
    r_out = np.zeros((tdim,len(lats_kanto),len(lons_kanto)))
    for i in range(tdim):
        # Apply gaussian filter (Smoothing)
        sigma = [2, 2] # smooth 250m scale to 1km scale
        r_sm = scipy.ndimage.filters.gaussian_filter(r_rect[i,:,:], sigma, mode='constant')
        # Interpolate by nearest neighbour
        intfunc = RegularGridInterpolator((lats_rect, lons_rect), r_sm)
        la2, lo2 = np.meshgrid(lats_kanto, lons_kanto)
        pts = np.vstack([la2.flatten(),lo2.flatten()])
        r_interp = intfunc(pts.T)
        r_interp = r_interp.reshape([len(lats_kanto),len(lons_kanto)]).T
        r_out[i,:,:] = r_interp
    return r_out

if __name__ == '__main__':
    
    lons_kanto, lats_kanto = grid_Kanto_jma_radar()

    # read
    infile_root = '../data/4p-hrncstprate/'
    print('input dir:',infile_root)

    # outfile
    outfile_root = '../data/hrncst_kanto/'
    print('output dir:',infile_root)

    nx = 200
    ny = 200
    nt = 7

    for infile in sorted(glob.iglob(infile_root + '/*00utc.nc.gz')):
        # read 1hour data at a time
        # initialize with -999.0
        R1h = np.full((nt,nx,ny),-999.0,dtype=np.float32)
    
        in_zfile = infile
        print('reading zipped file:',in_zfile)
        # '-k' option for avoiding removing gz file
        subprocess.run('gunzip -kf '+in_zfile,shell=True)
        in_nc=in_zfile.replace('.gz','')
        print('reading nc file:',in_nc)
        if os.path.exists(in_nc):
            R1h = read_hrncst_smoothed(lons_kanto,lats_kanto,in_nc)
        else:
            print('nc file not found!!!',in_nc)
        subprocess.run('rm '+in_nc,shell=True)
        # write to h5 file
        h5fname = infile.split('/')[-1]
        h5fname = h5fname.replace('.nc.gz','.h5')
        print('writing h5 file:',h5fname)
        h5file = h5py.File(outfile_root+h5fname,'w')
        h5file.create_dataset('R',data= R1h)
        import pdb; pdb.set_trace()





