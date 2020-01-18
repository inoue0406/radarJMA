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
    # trainning mode
    model.train()

    losses = AverageMeter()
    
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),sample_batched['future'].size())
        input = Variable(scl.fwd(sample_batched['features'].float())).cuda()
        past = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input, past, target)
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
        input = Variable(scl.fwd(sample_batched['features'].float())).cuda()
        past = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        # Forward
        # validation and testing process SHOULD NOT USE teacher forcing
        output = model(input, past, target, teacher_forcing_ratio = 0.0)
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

def test_epoch(test_loader,model,loss_fn,opt,scl):
    print('Testing for the model')
    
    # initialize
    RMSE_all = np.empty((0,opt.tdim_use),float)
    Xpast_all = np.empty((0,opt.tdim_use),float)
    Xtrue_all = np.empty((0,opt.tdim_use),float)
    Xmodel_all = np.empty((0,opt.tdim_use),float)

    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(test_loader):
        input = Variable(scl.fwd(sample_batched['features'].float())).cuda()
        past = Variable(scl.fwd(sample_batched['past'].float())).cuda()
        target = Variable(scl.fwd(sample_batched['future'].float())).cuda()
        # Forward
        # validation and testing process SHOULD NOT USE teacher forcing
        output = model(input, past, target, teacher_forcing_ratio = 0.0)
        loss = loss_fn(output, target)
        # concat all prediction data
        Xtrue = scl.inv(target.data.cpu().numpy().squeeze())
        Xmodel = scl.inv(output.data.cpu().numpy().squeeze())
        #
        Xpast_all = np.append(Xpast_all,past.squeeze(),axis=0)
        Xtrue_all = np.append(Xtrue_all,Xtrue,axis=0)
        Xmodel_all = np.append(Xmodel_all,Xmodel,axis=0)
        
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Test Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))

    # prep csv
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    RMSE = np.sqrt(np.mean((Xtrue_all-Xmodel_all)**2,axis=0))
    # import pdb; pdb.set_trace()
    df_out = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE})
    
    # apply eval metric by rain level
    levels = np.arange(-10,220,10)
    for i in range(len(levels)-1):
        low = levels[i]
        high = levels[i+1]
        id_range = (Xpast_all[:,-1] > low) * (Xpast_all[:,-1] <= high)
        print("range: ",low,high,"number of samples: ",np.sum(id_range))
        # calc rmse
        xt = Xtrue_all[id_range,:]
        xm = Xmodel_all[id_range,:]
        # RMSE along "samples" axis and keep time dim
        rr = np.sqrt(np.mean((xt-xm)**2,axis=0))
        vname = "RMSE_%d_%d" % (low,high)
        df_out[vname] = rr

    # save evaluated metric as csv file
    df_out.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime.csv'), float_format='%.3f')
    # free gpu memory
    del input,target,output,loss

    
