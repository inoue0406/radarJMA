import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from jma_pytorch_dataset import *
from utils import AverageMeter, Logger
from criteria_precip import *
# for debug
from tools_mem import *

# training/validation for one epoch

# --------------------------
# Training
# --------------------------

def train_epoch(epoch,num_epochs,train_loader,model,loss_fn,optimizer,train_logger,train_batch_logger,opt,scl):
    
    print('train at epoch {}'.format(epoch))

    losses = AverageMeter()
    
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),sample_batched['future'].size())
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # for logging
        losses.update(loss.item(), input.size(0))

        print('chk lr ',optimizer.param_groups[0]['lr'])
        train_batch_logger.log({
            'epoch': epoch,
            'batch': i_batch+1,
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if (i_batch+1) % 1 == 0:
            print ('Train Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch, num_epochs, i_batch+1, len(train_loader.dataset)//train_loader.batch_size, loss.item()))

    # update lr for optimizer
    optimizer.step()

    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    # free gpu memory
    del input,target,output,loss

# --------------------------
# Validation
# --------------------------

def valid_epoch(epoch,num_epochs,valid_loader,model,loss_fn,valid_logger,opt,scl):
    print('validation at epoch {}'.format(epoch))
    
    losses = AverageMeter()
        
    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(valid_loader):
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        
        # Forward
        output = model(input)
        loss = loss_fn(output, target)

        # for logging
        losses.update(loss.item(), input.size(0))

        if (i_batch+1) % 1 == 0:
            print ('Valid Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch, num_epochs, i_batch+1, len(valid_loader.dataset)//valid_loader.batch_size, loss.item()))
            
    valid_logger.log({
        'epoch': epoch,
        'loss': losses.avg})
    # free gpu memory
    del input,target,output,loss

# --------------------------
# Test
# --------------------------

def test_CLSTM_EP(test_loader,model,loss_fn,opt,scl,threshold):
    print('Testing for the model')
    
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

    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(test_loader):
        input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        
        # Forward
        output = model(input)
        loss = loss_fn(output, target)
        
        # apply evaluation metric
        Xtrue = scl.inv(target.data.cpu().numpy())
        Xmodel = scl.inv(output.data.cpu().numpy())
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(Xtrue,Xmodel,
                                                                  th=threshold)
        FSS_t = FSS_for_tensor(Xtrue,Xmodel,th=threshold,win=10)
        
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
            print ('Test Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))
    # logging for epoch-averaged loss
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
    df.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime_%.2f.csv' % threshold), float_format='%.3f')
    # free gpu memory
    del input,target,output,loss
    
