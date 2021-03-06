import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from jma_pytorch_dataset import *
from utils import AverageMeter, Logger
from criteria_precip import *

# testing for "Persistence" forecast 
            
def test_persistence(test_loader,loss_fn,test_logger,opt,scl,threshold):
    print('Test for persistence forecast')
    
    losses = AverageMeter()
    
    # initialize
    SumSE_all = np.empty((0,opt.tdim_use),float)
    hit_all = np.empty((0,opt.tdim_use),float)
    miss_all = np.empty((0,opt.tdim_use),float)
    falarm_all = np.empty((0,opt.tdim_use),float)
    m_xy_all = np.empty((0,opt.tdim_use),float)
    m_xx_all = np.empty((0,opt.tdim_use),float)
    m_yy_all = np.empty((0,opt.tdim_use),float)
    MaxSE_all = np.empty((0,opt.tdim_use),float)
    FSS_t_all = np.empty((0,opt.tdim_use),float)

    for i_batch, sample_batched in enumerate(test_loader):
        input = Variable(scl.fwd(sample_batched['past'].float())).cpu()
        target = Variable(scl.fwd(sample_batched['future'].float())).cpu()
        
        # Prediction by Persistence
        output = target.clone()
        #for n in range(test_loader.batch_size):
        for it in range(opt.tdim_use):
            # predict by the latest frame
            output.data[:,it,0,:,:] = input.data[:,(opt.tdim_use-1),0,:,:]
            # zero prediction
            #output.data[:,it,0,:,:] = 0.0
    
        loss = loss_fn(output, target)

        # for logging
        losses.update(loss.item(), input.size(0))
        
        # apply evaluation metric
        Xtrue = scl.inv(target.data.cpu().numpy())
        Xmodel = scl.inv(output.data.cpu().numpy())
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(Xtrue,Xmodel,th=threshold)
        FSS_t = FSS_for_tensor(Xtrue,Xmodel,th=threshold,win=10)
        # stat
        SumSE_all = np.append(SumSE_all,SumSE,axis=0)
        hit_all = np.append(hit_all,hit,axis=0)
        miss_all = np.append(miss_all,miss,axis=0)
        falarm_all = np.append(falarm_all,falarm,axis=0)
        m_xy_all = np.append(m_xy_all,m_xy,axis=0)
        m_xx_all = np.append(m_xx_all,m_xx,axis=0)
        m_yy_all = np.append(m_yy_all,m_yy,axis=0)
        MaxSE_all = np.append(MaxSE_all,MaxSE,axis=0)
        FSS_t_all = np.append(FSS_t_all,FSS_t,axis=0)
        
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Testing, Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))
    # logging for averaged loss
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,
                                          MaxSE_all,FSS_t_all,axis=None)
    test_logger.log({
        'loss': losses.avg,
        'RMSE': RMSE,
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'Cor': Cor,
        'MaxMSE': MaxMSE,
        'FSS_mean': FSS_mean,
        })
    # logging for loss by time
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                          m_xy_all,m_xx_all,m_yy_all,
                                          MaxSE_all,FSS_t_all,axis=(0))
    # save evaluated metric as csv file
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    # import pdb; pdb.set_trace()
    df = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE,
                       'CSI':CSI,
                       'FAR':FAR,
                       'POD':POD,
                       'Cor':Cor,
                       'MaxMSE': MaxMSE,
                       'FSS_mean': FSS_mean,
                       })
    df.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime_%.2f.csv' % threshold), float_format='%.3f')

    
