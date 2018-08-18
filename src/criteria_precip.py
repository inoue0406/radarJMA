# 
# several criteria for evaluating precipitation 
# 

import numpy as np
import pdb

# Various statistics for evaluation of rainfall
def StatRainfall(Xtrue,Xmodel,th=0.5):
    # count only if rainfall is present
    rflag = Xtrue > th
    # ----------------------
    # calc rainfall SE
    Xt = rflag * Xtrue
    Xm = rflag * Xmodel
    SE = np.power(Xt - Xm,2)
    SumSE = np.sum(SE,axis=(2,3,4))
    # ----------------------
    # calc hit,miss,falarm
    flg_tr = Xtrue > th
    flg_mo = Xmodel > th
    # cal hit/miss/falsealarm
    hit    = np.sum(flg_tr * flg_mo     ,axis=(2,3,4))
    miss   = np.sum(flg_tr * (1-flg_mo) ,axis=(2,3,4))
    falarm = np.sum((1-flg_tr) * flg_mo ,axis=(2,3,4))
    # ----------------------1
    # calc moments for correlation
    m_xy     = np.sum(Xtrue*Xmodel ,axis=(2,3,4))
    m_xx     = np.sum(Xtrue*Xtrue  ,axis=(2,3,4))
    m_yy     = np.sum(Xmodel*Xmodel,axis=(2,3,4))
    return(SumSE,hit,miss,falarm,m_xy,m_xx,m_yy)

def MetricRainfall(SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,axis=None):
    # "axis=(0,1)" produces stat of [sample x time] dim
    # "axis=None" produces scalar statistics for the whole batch
    # 
    # reduce dim if specified
    SumSE  = np.sum(SumSE,axis=axis)
    hit    = np.sum(hit,axis=axis)
    miss   = np.sum(miss,axis=axis)
    falarm = np.sum(falarm,axis=axis)
    m_xy   = np.sum(m_xy,axis=axis)
    m_xx   = np.sum(m_xx,axis=axis)
    m_yy   = np.sum(m_yy,axis=axis)
    # calc metrics based on statistics
    Nrain = hit+miss
    # Rainfall MSE
    is_zero = (Nrain==0) 
    Nrain = Nrain*(1-is_zero) + 9999*(is_zero) # for avoiding zero division
    RMSE = SumSE/Nrain
    # calc CSI
    denom = hit+miss+falarm
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set CSI as 0
    CSI = hit/denom
    # calc FAR
    denom = hit+falarm
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set FAR as 0
    FAR = falarm/denom
    # calc POD
    denom = hit+miss
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set POD as 0
    POD = hit/denom
    # calc Cor
    eps = 0.0001 # small
    Cor = m_xy/(np.sqrt(m_xx*m_yy)+eps)
    return(RMSE,CSI,FAR,POD,Cor)

# Rainfall MSE
def RainfallMSE(Xtrue,Xmodel,th=0.5):
    # The routine expects tensors to be in the order of
    # "batch,tsize,layer,width,height"
    #
    # count only if rainfall is present
    rflag = Xtrue > th
    # calc rainfall mse
    Xt = rflag * Xtrue
    Xm = rflag * Xmodel
    SE = np.power(Xt - Xm,2)
    # count only for nonzero elements
    N = np.sum(rflag,axis=(2,3,4))
    is_zero = (N==0) 
    N = N*(1-is_zero) + 9999*(is_zero) # for avoiding zero division
    MSE = np.sum(SE,axis=(2,3,4))/N
    return(MSE)

# Critical Success Index (CSI) 
def CSI(Xtrue,Xmodel,th=0.5):
    # count only if rainfall is present
    flg_tr = Xtrue > th
    flg_mo = Xmodel > th
    # cal hit/miss/falsealarm
    hit    = np.sum(flg_tr * flg_mo     ,axis=(2,3,4))
    miss   = np.sum(flg_tr * (1-flg_mo) ,axis=(2,3,4))
    falarm = np.sum((1-flg_tr) * flg_mo ,axis=(2,3,4))
    # calc CSI
    denom = hit+miss+falarm
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set CSI as 0
    CSI = hit/denom
    return(CSI)

# False Alarm Rate (FAR)
def FAR(Xtrue,Xmodel,th=0.5):
    # count only if rainfall is present
    flg_tr = Xtrue > th
    flg_mo = Xmodel > th
    # cal hit/miss/falsealarm
    hit    = np.sum(flg_tr * flg_mo     ,axis=(2,3,4))
    falarm = np.sum((1-flg_tr) * flg_mo ,axis=(2,3,4))
    # calc FAR
    denom = hit+falarm
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set FAR as 0
    FAR = falarm/denom
    return(FAR)

# Probability of detection (POD)
def POD(Xtrue,Xmodel,th=0.5):
    # count only if rainfall is present
    flg_tr = Xtrue > th
    flg_mo = Xmodel > th
    # cal hit/miss/falsealarm
    hit    = np.sum(flg_tr * flg_mo     ,axis=(2,3,4))
    miss   = np.sum(flg_tr * (1-flg_mo) ,axis=(2,3,4))
    # calc POD
    denom = hit+miss
    is_zero = (denom==0)
    denom = denom + is_zero*9999.0 # if zero, set POD as 0
    POD = hit/denom
    return(POD)

# Correlation
def Correlation(Xtrue,Xmodel):
    # calc moments
    m_xy     = np.sum(Xtrue*Xmodel ,axis=(2,3,4))
    m_xx     = np.sum(Xtrue*Xtrue  ,axis=(2,3,4))
    m_yy     = np.sum(Xmodel*Xmodel,axis=(2,3,4))
    # calc Correlation
    eps = 0.0001 # small
    Cor = m_xy/(np.sqrt(m_xx*m_yy)+eps)
    return(Cor)

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
    Xmodel = Xtrue + Xmodel*0.2
    
    #MSE = RainfallMSE(Xtrue,Xmodel)
    #CSI = CSI(Xtrue,Xmodel)
    #FAR = FAR(Xtrue,Xmodel)
    #POD = POD(Xtrue,Xmodel)
    #Cor = Correlation(Xtrue,Xmodel)
    
    #SumSE,hit,miss,falarm,m_xy,m_xx,m_yy = StatRainfall(Xtrue,Xmodel,th=0.5)
    
    SumSE,hit,miss,falarm,m_xy,m_xx,m_yy = StatRainfall(Xtrue,Xmodel,th=0.5)
    RMSE,CSI,FAR,POD,Cor = MetricRainfall(SumSE,hit,miss,falarm,m_xy,m_xx,m_yy)
    
    pdb.set_trace()


