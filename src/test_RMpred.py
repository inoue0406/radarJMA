import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from jma_pytorch_dataset import *
from convolution_lstm_mod import *
from utils import AverageMeter, Logger
from criteria_precip import *

import sys

# optical flow forecast with RainyMotion

# add rainymotion source dir
path = os.path.join('../../rainymotion')
sys.path.append(path)
from rainymotion.models import *
from rainymotion.utils import *

def RM_predictor(data_past,tdim_use):
    # prediction by rainymotion
    # "Dense" variant of model is used
    
    # data_past : tensor with dimension of [tsize,width,height]
    xmx = data_past.max() # max value
    eps = 1.0e-3
    if(xmx < eps):
        # for events with no rain, simply return zero tensor
        return(data_past) 
    # initialize the model
    model = Dense()
    model.lead_steps = tdim_use
    # scale by log transform
    data_scaled,c1,c2 = RYScaler(data_past)
    c2 = max(c2,1.0) # c2 should be larger than 1.0
    #print('scaling params:',c1,c2)
    # upload data to the model instance
    model.input_data = data_scaled
    # run the model with default parameters
    nowcast = model.run()
    # inverse scaling
    nowcast_orig = inv_RYScaler(nowcast,c1,c2)
    return(nowcast_orig)
            
def test_RMpred(test_loader,loss_fn,test_logger,opt):
    print('Test for RainyMotion optical-flow forecast')
    
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
        input = Variable(sample_batched['past']).cpu() 
        target = Variable(sample_batched['future']).cpu()
         # note that no regularization is needed here, since it is taken care by RM_predictor

        print('batch:',i_batch,'\n')
        # Prediction by Persistence
        output = target.clone()
        for n in range(input.data.shape[0]):
            print('past file name:',n,sample_batched['fnames_past'][n])
            output.data[n,:,0,:,:] = torch.from_numpy(RM_predictor(input.data.numpy()[n,:,0,:,:],opt.tdim_use))
            
        loss = loss_fn(output, target)

        # for logging
        losses.update(loss.item(), input.size(0))
        
        # apply evaluation metric
        Xtrue = target.data.cpu().numpy()
        Xmodel = output.data.cpu().numpy()
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(target.data.cpu().numpy(),
                                                            output.data.cpu().numpy(),
                                                            th=opt.eval_threshold)
        FSS_t = FSS_for_tensor(Xtrue,Xmodel,th=opt.eval_threshold,win=10)
        
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
        'FSS_mean': FSS_mean})

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
                       'FSS_mean': FSS_mean})
    df.to_csv(os.path.join(opt.result_path, 'test_evaluation_predtime.csv'))


    
