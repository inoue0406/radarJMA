import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

# for optical flow
from scipy import interpolate
import cv2

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from utils import AverageMeter, Logger
from criteria_precip import *

# testing for Optical Flow forecast

def OF_predictor(data_past):
    # data_past : tensor with dimension of [tsize,width,height]
    # define xy grid
    #data_past = np.transpose(data_past,(0,2,1)) #TEMP
    tsize = data_past.shape[0]
    xsize = data_past.shape[1]
    ysize = data_past.shape[2]
    data_nxt = np.zeros((tsize,xsize,ysize))
    xgrd = np.arange(0,xsize)
    ygrd = np.arange(0,ysize)
    X, Y = np.meshgrid(xgrd,ygrd)
    #-------------------------------------
    # Step1 : Get flow vector by optical flow
    ofsize = 6
    flow_all = np.zeros((ofsize,xsize,ysize,2))
    for i in range((tsize-1-ofsize),(tsize-1)):
        #print('calc OF between ',i,' and ',i+1)
        flow_all[i-5,:,:,:] = cv2.calcOpticalFlowFarneback(data_past[i,:,:],data_past[i+1,:,:],None,
                                                         pyr_scale=0.5,levels=2,winsize=10,iterations=3,
                                                         poly_n=5,poly_sigma=1.1,flags=False)
        #flow_all[i-5,:,:,:] = cv2.calcOpticalFlowFarneback(data_past[0,:,:],data_past[tsize-1,:,:],None,
        #                                                 pyr_scale=0.5,levels=2,winsize=10,iterations=3,
        #                                                 poly_n=5,poly_sigma=1.1,flags=False)
    # take median as representative OF
    medflow = np.median(flow_all,axis=0)
    # adjust absolute velocity of the flow by constant
    # coeff = 1.0e6
    coeff = 1.0
    medflow = medflow * (-1) * coeff
    #-------------------------------------
    # Step2 : Time-evolution by semi-Lagrangian method
    # x(n) = x(n-1) + dt*v
    # -> x(n-1) = x(n) - dt*v
    VX = medflow[:,:,0]
    VY = medflow[:,:,1]
    print('estimated velocity: vx median=',np.median(VX),', vy median=',np.median(VY))
    print('estimated velocity: vx min/max=',np.min(VX),np.max(VX),', vy min/max=',np.min(VY),np.max(VY))
    #import pdb;pdb.set_trace()
    # prep interp func
    R = data_past[tsize-1,:,:] # use last picture as initial condition
    itp_rain = interpolate.RectBivariateSpline(xgrd, ygrd, R)
    Rp = data_past[tsize-2,:,:] # previous step for change ratio calculation
    itp_rain_p = interpolate.RectBivariateSpline(xgrd, ygrd, Rp)

    # calc change ratio along with time evolution
    dt = 1.0
    xp1 = X - dt*VX
    yp1 = Y - dt*VY
    xp2 = X - 2.0*dt*VX
    yp2 = Y - 2.0*dt*VY
    # apply interpolation
    Rp1 = itp_rain.ev(xp1,yp1)
    Rp2 = itp_rain_p.ev(xp2,yp2)
    flg_r = Rp2 > 0.5 # rain existence flag
    eps = 0.001
    # if no rain, set ratio=1.0
    ratio = flg_r*Rp1/(Rp2+eps) + (1.0-flg_r)*1.0
    for it in range(tsize):
        dt = 1.0*(it+1)
        xprev = X - dt*VX
        yprev = Y - dt*VY
        # apply interpolation
        data_nxt[it,:,:] = itp_rain.ev(xprev,yprev)#*np.power(ratio,(it+1))
        #import pdb; pdb.set_trace()
        # TEMP for chk
        #data_nxt[it,:,:] = data[tsize-1,:,:]
    return(data_nxt)
            
def test_OFpred(test_loader,loss_fn,test_logger,opt):
    print('Test for Optical Flow forecast')
    
    losses = AverageMeter()
    
    # initialize
    SumSE_all = np.empty((0,opt.tdim_use),float)
    hit_all = np.empty((0,opt.tdim_use),float)
    miss_all = np.empty((0,opt.tdim_use),float)
    falarm_all = np.empty((0,opt.tdim_use),float)
    m_xy_all = np.empty((0,opt.tdim_use),float)
    m_xx_all = np.empty((0,opt.tdim_use),float)
    m_yy_all = np.empty((0,opt.tdim_use),float)

    for i_batch, sample_batched in enumerate(test_loader):
        input = Variable(sample_batched['past']).cpu()
        target = Variable(sample_batched['future']).cpu()

        print('batch:',i_batch,'\n')
        # Prediction by Persistence
        output = target.clone()
        for n in range(input.data.shape[0]):
            output.data[n,:,0,:,:] = torch.from_numpy(OF_predictor(input.data.numpy()[n,:,0,:,:]))
            
        loss = loss_fn(output, target)

        # for logging
        losses.update(loss.data[0], input.size(0))
        
        # apply evaluation metric
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy = StatRainfall(target.data.cpu().numpy()*201.0,
                                                            output.data.cpu().numpy()*201.0,
                                                            th=0.5)
        SumSE_all = np.append(SumSE_all,SumSE,axis=0)
        hit_all = np.append(hit_all,hit,axis=0)
        miss_all = np.append(miss_all,miss,axis=0)
        falarm_all = np.append(falarm_all,falarm,axis=0)
        m_xy_all = np.append(m_xy_all,m_xy,axis=0)
        m_xx_all = np.append(m_xx_all,m_xx,axis=0)
        m_yy_all = np.append(m_yy_all,m_yy,axis=0)
        
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Testing, Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.data[0]))
    # logging for averaged loss
    RMSE,CSI,FAR,POD,Cor = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,axis=None)
    test_logger.log({
        'loss': losses.avg,
        'RMSE': RMSE,
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'Cor': Cor})
    # logging for loss by time
    RMSE,CSI,FAR,POD,Cor = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,axis=(0))
    # save evaluated metric as csv file
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    # import pdb; pdb.set_trace()
    df = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE,
                       'CSI':CSI,
                       'FAR':FAR,
                       'POD':POD,
                       'Cor':Cor})
    df.to_csv(os.path.join(opt.result_path, 'test_evaluation_predtime.csv'))


if __name__ == '__main__':
    #
    # test code for OF_predictor
    #
    import h5py
    # test with artificial sample data
    tsize = 12
    xsize = 200
    ysize = 200
    
    rmax = 201.0 # max rainfall intensity    
    h5file = h5py.File('../data/data_h5/2p-jmaradar5_2017-08-01_1100utc.h5','r')
    rain_X = h5file['R'].value/rmax

    #import pdb;pdb.set_trace()
    
    # circular shape
    # radius = 10
    #xgrd = np.arange(0,xsize)
    #ygrd = np.arange(0,ysize)
    #X, Y = np.meshgrid(xgrd,ygrd)
    #R = np.zeros((xsize,ysize))
    #in_circle = (np.power(X-xsize/2.0,2)+np.power(Y-ysize/2.0,2)) < radius**2
    #R = R + 1.0*in_circle
    R = rain_X[tsize-1,:,:]

    # generate artificial data
    for k in range(3):
        print('moving in x direction k=',k)
        data_past = np.zeros((tsize,xsize,ysize))
        data_future = np.zeros((tsize,xsize,ysize))
        for i in range(tsize):
            data_past[i,:,:] = np.roll(R,i*(k+1),axis=0)
        pred_future = OF_predictor(data_past)
        
    for k in range(3):
        print('moving in y direction k=',k)
        data_past = np.zeros((tsize,xsize,ysize))
        data_future = np.zeros((tsize,xsize,ysize))
        for i in range(tsize):
            data_past[i,:,:] = np.roll(R,i*(k+1),axis=1)
        pred_future = OF_predictor(data_past)

    # apply the trained model to the data
#    input = Variable(sample_batched['past']).cpu()
#    target = Variable(sample_batched['future']).cpu()
#    # prediction by optical flow
#    output = target.clone()
#    for n in range(input.data.shape[0]):
#        output.data[n,:,0,:,:] = OF_predictor(input.data[n,:,0,:,:])
#    #

   
    

    
