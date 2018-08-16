import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from utils import AverageMeter, Logger
from criteria_precip import *

# validation for "Persistence" forecast 
            
def valid_persistence_epoch(epoch,num_epochs,valid_loader,loss_fn,valid_logger,opt):
    print('validation at epoch {}'.format(epoch+1))
    
    losses = AverageMeter()
    
    # initialize
    SumSE_all = np.empty((0,opt.tdim_use),float)
    hit_all = np.empty((0,opt.tdim_use),float)
    miss_all = np.empty((0,opt.tdim_use),float)
    falarm_all = np.empty((0,opt.tdim_use),float)
    m_xy_all = np.empty((0,opt.tdim_use),float)
    m_xx_all = np.empty((0,opt.tdim_use),float)
    m_yy_all = np.empty((0,opt.tdim_use),float)

    for i_batch, sample_batched in enumerate(valid_loader):
        input = Variable(sample_batched['past']).cpu()
        target = Variable(sample_batched['future']).cpu()
        
        # Prediction by Persistence
        output = target.clone()
        #for n in range(valid_loader.batch_size):
        for it in range(opt.tdim_use):
            # predict by the latest frame
            # output.data[:,it,0,:,:] = input.data[:,(opt.tdim_use-1),0,:,:]
            # zero prediction
            output.data[:,it,0,:,:] = 0.0
    
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
            print ('Valid Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch+1, num_epochs, i_batch+1, len(valid_loader.dataset)//valid_loader.batch_size, loss.data[0]))
    # logging for epoch-averaged loss
    RMSE,CSI,FAR,POD,Cor = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,reduce=True)
    valid_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'RMSE': RMSE,
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'Cor': Cor, 
    })

    
