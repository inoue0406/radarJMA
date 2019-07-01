import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.nn import functional as F

from jma_pytorch_dataset import *
from regularizer import *
from convolution_lstm_mod import *
from utils import AverageMeter, Logger
from criteria_precip import *

# training/validation for one epoch


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# --------------------------
# Training
# --------------------------

def train_epoch(epoch,num_epochs,train_loader,model,optimizer,train_logger,train_batch_logger,opt,reg):
    
    print('train at epoch {}'.format(epoch+1))

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
    
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),
        #    sample_batched['future'].size())
        # for VAE, concatenate past&future data
        sample_all = torch.cat([sample_batched['past'][:,:,0,:,:],
                                sample_batched['future'][:,:,0,:,:]],
                               dim=1)
        in_vae = Variable(reg.fwd(sample_all)).cuda()
        #
        in_lstm = Variable(reg.fwd(sample_batched['past'])).cuda()
        target = Variable(reg.fwd(sample_batched['future'])).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output, mu, logvar = model(in_vae,in_lstm)
        loss = loss_function(output, target, mu, logvar)
        loss.backward()
        optimizer.step()
        # for logging
        losses.update(loss.item(), in_lstm.size(0))
        # apply evaluation metric
        Xtrue = reg.inv(target.data.cpu().numpy())
        Xmodel = reg.inv(output.data.cpu().numpy())
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(Xtrue,Xmodel,
                                                                  th=opt.eval_threshold)
        FSS_t = FSS_for_tensor(Xtrue,Xmodel,th=opt.eval_threshold,win=10)

        RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE,hit,miss,falarm,
                                                              m_xy,m_xx,m_yy,
                                                              MaxSE_all,FSS_t_all,axis=None)

        SumSE_all = np.append(SumSE_all,SumSE,axis=0)
        hit_all = np.append(hit_all,hit,axis=0)
        miss_all = np.append(miss_all,miss,axis=0)
        falarm_all = np.append(falarm_all,falarm,axis=0)
        m_xy_all = np.append(m_xy_all,m_xy,axis=0)
        m_xx_all = np.append(m_xx_all,m_xx,axis=0)
        m_yy_all = np.append(m_yy_all,m_yy,axis=0)
        MaxSE_all = np.append(MaxSE_all,MaxSE,axis=0)
        FSS_t_all = np.append(FSS_t_all,FSS_t,axis=0)
        
        print('chk lr ',optimizer.param_groups[0]['lr'])
        train_batch_logger.log({
            'epoch': epoch+1,
            'batch': i_batch+1,
            'loss': losses.val,
            'RMSE': RMSE,
            'CSI': CSI,
            'FAR': FAR,
            'POD': POD,
            'Cor': Cor, 
            'MaxMSE': MaxMSE,
            'FSS_mean': FSS_mean,
            'lr': optimizer.param_groups[0]['lr']
        })
        # 
        if (i_batch+1) % 1 == 0:
            print ('Train Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch+1, num_epochs, i_batch+1, len(train_loader.dataset)//train_loader.batch_size, loss.item()))

    # update lr for optimizer
    optimizer.step()
    
    # logging for epoch-averaged loss
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                                          m_xy_all,m_xx_all,m_yy_all,
                                                          MaxSE_all,FSS_t_all,axis=None)
    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'RMSE': RMSE,
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'Cor': Cor,
        'MaxMSE': MaxMSE,
        'FSS_mean': FSS_mean,
        'lr': optimizer.param_groups[0]['lr']
    })
    # free gpu memory
    del in_vae,in_lstm,target,output,loss
    
# --------------------------
# Validation
# --------------------------

def valid_epoch(epoch,num_epochs,valid_loader,model,valid_logger,opt,reg):
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
    MaxSE_all = np.empty((0,opt.tdim_use),float)
    FSS_t_all = np.empty((0,opt.tdim_use),float)

    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(valid_loader):
        # put zero for future data
        sample_all = torch.cat([sample_batched['past'][:,:,0,:,:],
                                torch.zeros(sample_batched['past'][:,:,0,:,:].shape)],
                               dim=1)
        in_vae = Variable(reg.fwd(sample_all)).cuda()
        in_lstm = Variable(reg.fwd(sample_batched['past'])).cuda()
        target = Variable(reg.fwd(sample_batched['future'])).cuda()
        
        # Forward
        output, mu, logvar = model(in_vae,in_lstm)
        loss = loss_function(output, target, mu, logvar)

        # for logging
        losses.update(loss.item(), in_lstm.size(0))
        
        # apply evaluation metric
        Xtrue = reg.inv(target.data.cpu().numpy())
        Xmodel = reg.inv(output.data.cpu().numpy())
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(Xtrue,Xmodel,
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
            print ('Valid Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch+1, num_epochs, i_batch+1, len(valid_loader.dataset)//valid_loader.batch_size, loss.item()))
    # logging for epoch-averaged loss
    RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                                          m_xy_all,m_xx_all,m_yy_all,
                                                          MaxSE_all,FSS_t_all,axis=None)
    valid_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'RMSE': RMSE,
        'CSI': CSI,
        'FAR': FAR,
        'POD': POD,
        'Cor': Cor, 
        'MaxMSE': MaxMSE,
        'FSS_mean': FSS_mean})
    # free gpu memory
    del in_vae,in_lstm,target,output,loss


# --------------------------
# Test
# --------------------------

def test_CLSTM_EP(test_loader,model,opt,reg):
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
        # put zero for future data
        sample_all = torch.cat([sample_batched['past'][:,:,0,:,:],
                                torch.zeros(sample_batched['past'][:,:,0,:,:].shape)],
                               dim=1)
        in_vae = Variable(reg.fwd(sample_all)).cuda()
        in_lstm = Variable(reg.fwd(sample_batched['past'])).cuda()
        target = Variable(reg.fwd(sample_batched['future'])).cuda()
        
        # Forward
        output, mu, logvar = model(in_vae,in_lstm)
        loss = loss_function(output, target, mu, logvar)
        
        # apply evaluation metric
        Xtrue = reg.inv(target.data.cpu().numpy())
        Xmodel = reg.inv(output.data.cpu().numpy())
        SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(Xtrue,Xmodel,
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
                           'test_evaluation_predtime_%.2f.csv' % opt.eval_threshold))
    # free gpu memory
    del in_vae,in_lstm,target,output,loss

    


    
