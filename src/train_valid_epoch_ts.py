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
        output = model(input, target)
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

