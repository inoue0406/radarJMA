# 
# FSS computation code
# taken from
# https://github.com/nathan-eize/fss/blob/master/fss_core.py
# 

#--------------------------------------------
# FSS definitions
#--------------------------------------------
# Added content
# Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.

import collections
import scipy.signal as signal
import numpy as np
import pandas as pd

# Define an integral image data-type.
IntegralImage = collections.namedtuple(
    'IntegralImage',
    'table padding'
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def summedAreaTable(field, padding=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Returns the summed-area-table of the provided field.
    """
    if padding is None:
        return IntegralImage(
            table=field.cumsum(1).cumsum(0),
            padding=0
        )
    else:
        # Zero pad and account for windows centered on boundaries
        expandedField = np.zeros(
            (field.shape[0] + (2 * padding),
             field.shape[1] + (2 * padding))
        )
        expandedField[
            padding:padding + field.shape[0],
            padding:padding + field.shape[1]] = field
        # Compute the summed area table.
        return IntegralImage(
            table=expandedField.cumsum(1).cumsum(0),
            padding=padding
        )
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
def integralField(field, n, integral=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Fast summed area table version of the sliding accumulator.
    @param field: nd-array of binary hits/misses.
    @param n: window size.
    """
    window = int(n / 2)
    
    if integral is not None:
        
        assert integral.padding >= window, 'Expected larger table.'
        
        integral = IntegralImage(
            table=integral.table[
                (integral.padding - window):-(integral.padding - window),
                (integral.padding - window):-(integral.padding - window)
            ],
            padding=window
        )
        
    else:
        integral = summedAreaTable(field, padding=window)
        
    # Compute the coordinates of the grid, offset by the window size.
    gridcells = np.mgrid[0:field.shape[0], 0:field.shape[1]] + integral.padding
    
    tl = (gridcells - integral.padding) - 1
    br = (gridcells + integral.padding - 1)
    
    sumField = integral.table[br[0], br[1]] + \
               integral.table[tl[0], tl[1]] - \
               integral.table[tl[0], br[1]] - \
               integral.table[br[0], tl[1]]
    
    # Fix values on the top and left boundaries of the field.
    sumField[:, 0] = integral.table[br[0], br[1]][:, 0]
    sumField[0, :] = integral.table[br[0], br[1]][0, :]
    sumField[integral.padding:, 0] -= \
                                      integral.table[tl[0], br[1]][integral.padding:, 0]
    sumField[0, integral.padding:] -= \
                                      integral.table[br[0], tl[1]][0, integral.padding:]
    return sumField

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def calcFSS(fcst, obs, threshold, window, fcst_cache=None, obs_cache=None):
    """
    Code from Faggian N, Roux B, Steinle P, Ebert B. 2015. Fast calculation of the fractions skill score. Mausam 66: 457466.
    Compute the fraction skill score.
    @param fcst: nd_array, forecast field.
    @param obs: nd_array, observation field.
    @param method: python function, defining the smoothing method.
    @return: float, numerator, denominator and FSS
    FSS = 1 - numerator/denominator
    """
    fhat = integralField(fcst>threshold, window)
    ohat = integralField(obs>threshold, window)
    
    scale = 1.0 / fhat.size
    num = np.nanmean(np.power(fhat - ohat, 2))
    denom = np.nanmean(np.power(fhat, 2) + np.power(ohat, 2))
    
    #print('in calcFSS num,denom=',num,denom)
    eps = 1.0e-6
    if(denom < eps):
        # if denom is small, return NaN
        fss = np.nan
    else:
        fss = 1.-num/denom
    return num, denom, fss


if __name__ == '__main__':
    # test with random sample data
    batch = 10
    tsize = 6
    layer = 1
    width = 12
    height = 12
    #X <- np.zeros((width,height))
    Xtrue = np.random.rand(batch,tsize,layer,width,height)
    Xmodel = np.random.randn(batch,tsize,layer,width,height)
    
    Xtrue = np.abs(Xtrue)*0.60
    Xmodel = Xtrue + Xmodel*0.3
    #Xmodel = Xtrue

    #num,denom,fss = calcFSS(Xtrue[0,0,0,:,:],Xmodel[0,0,0,:,:],threshold=0.5,window=2)
    #print('num/denom/fss:',num,denom,fss)
    
    # chk for small data
    # to be checked with hand calculation
    Xt = np.array([[0,0,0,2,0],
                   [0,1,2,3,2],
                   [0,0,2,2,1],
                   [0,0,1,1,1],
                   [0,0,0,0,0]])
    Xm = np.array([[0,0,0,1,0],
                   [0,2,3,2,1],
                   [0,2,0,0,0],
                   [0,0,1,1,1],
                   [0,0,0,0,0]])
    num,denom,fss = calcFSS(Xt,Xm,threshold=1.0,window=2)
    print('num/denom/fss:',num,denom,fss)



