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

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_with_map(data,lons,lats,fname):
    '''
    Plot rainfall pattern with a map
    '''
    # Get some parameters for the Stereographic Projection
    lat_0 = 35.7
    lon_0 = 139.8
    m = Basemap(width=300000,height=300000,
            resolution='i',projection='stere',\
            lat_ts=35,lat_0=lat_0,lon_0=lon_0)
    # define coordinates
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    # Plot Data
    cs = m.pcolor(xi,yi,np.squeeze(data),vmax=30.0)
    # Add Grid Lines
    m.drawparallels(np.arange(0, 50., 1.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(100., 150., 1.), labels=[0,0,0,1], fontsize=10)
    # Add Coastlines
    m.drawcoastlines()
    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    # Add Title
    plt.title('Rainfall')

    #plt.show()
    plt.savefig(fname)

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
    delta = 1.0
    lon_min = np.min(lons_kanto) - delta
    lon_max = np.max(lons_kanto) + delta
    lat_min = np.min(lats_kanto) - delta
    lat_max = np.max(lats_kanto) + delta

    nc = netCDF4.Dataset('../data/work/jma_hrncst/4p-hrncstprate_japan0250_2017-08-01_1100utc.nc', 'r')
    # dimensions
    nx = len(nc.dimensions['LON'])
    ny = len(nc.dimensions['LAT'])
    nt = len(nc.dimensions['TIME'])
    print("dims:",nx,ny,nt)
    # extract variable
    lons = np.array(nc.variables['LON'][:])
    lats = np.array(nc.variables['LAT'][:])
    R = nc.variables['PRATE'][:] # numpy.ma.core.MaskedArr  ay
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
    # Apply gaussian filter (Smoothing)
    sigma = [2, 2] # smooth 250m scale to 1km scale
    r_sm = scipy.ndimage.filters.gaussian_filter(r_rect[0,:,:], sigma, mode='constant')
    plt.imshow(R[0,:,:])
    plt.savefig("alljapan.png")
    # Interpolate by nearest neighbour
    intfunc = RegularGridInterpolator((lats_rect, lons_rect), r_sm)
    la2, lo2 = np.meshgrid(lats_kanto, lons_kanto)
    pts = np.vstack([la2.flatten(),lo2.flatten()])
    r_interp = intfunc(pts.T)
    r_interp = r_interp.reshape([len(lats_kanto),len(lons_kanto)]).T 
    plot_with_map(r_interp,lons_kanto,lats_kanto,"smooth_interp_with_map.png")
    #plot_with_map(R[0,:,:],lons,lats,"alljapan_with_map.png")
    plt.imshow(r_rect[0,:,:])
    plt.savefig("smooth_before.png")
    plot_with_map(r_rect[0,:,:],lons_rect,lats_rect,"smooth_before_with_map.png")
    plt.imshow(r_sm)
    plt.savefig("smooth_after.png")
    plot_with_map(r_sm,lons_rect,lats_rect,"smooth_after_with_map.png")
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    
    lons_kanto, lats_kanto = grid_Kanto_jma_radar()
    read_hrncst_smoothed(lons_kanto,lats_kanto)
    import pdb;pdb.set_trace()




