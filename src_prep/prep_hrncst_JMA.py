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

if __name__ == '__main__':
    
    nc = netCDF4.Dataset('../data/jma_hrncst/4p-hrncstprate_japan0250_2017-01-01_0000utc.nc', 'r')
    import pdb;pdb.set_trace()



