import numpy as np
import torch 
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import hickle as hkl
import os
import sys
import json
import time

import pdb

from mnist_pytorch_dataset import *
from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *

import skimage.measure

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")

# test code for mnist
# calc MSE, MAE, SSIM
def test_CLSTM_EP_mnist(test_loader,model,loss_fn,opt,scl):
    print('Testing for the model')
    
    # initialize
    MSE_all = np.empty((0,opt.tdim_use),float)
    MAE_all = np.empty((0,opt.tdim_use),float)
    SSIM_all = np.empty((0,opt.tdim_use),float)

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

        # metric in (batch x time) dim
        MSE = ((Xmodel/255.0-Xtrue/255.0)**2).mean(axis=(2,3,4))
        MAE = (np.abs(Xmodel/255.0-Xtrue/255.0)).mean(axis=(2,3,4))
        # SSIM index
        SSIM = np.zeros((Xmodel.shape[0],Xmodel.shape[1]))
        for k in range(Xmodel.shape[0]):
            for l in range(Xmodel.shape[1]):
                SSIM[k,l] = skimage.measure.compare_ssim(Xmodel[k,l,0,:,:],Xtrue[k,l,0,:,:])
        
        MSE_all = np.append(MSE_all,MSE,axis=0)
        MAE_all = np.append(MAE_all,MAE,axis=0)
        SSIM_all = np.append(SSIM_all,SSIM,axis=0)
        
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Test Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))
            
    # save evaluated metric as csv file
    tpred = (np.arange(opt.tdim_use)+1.0)
    df = pd.DataFrame({'tpred_min':tpred,
                       'MSE':MSE_all.mean(axis=0),
                       'MAE':MAE_all.mean(axis=0),
                       'SSIM':SSIM_all.mean(axis=0)})
    df.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime.csv'))
    # free gpu memory
    del input,target,output,loss

    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(opt)
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')

    # prepare scaler for data
    scl = LinearScaler(rmax=255.0)
    
    if not opt.no_train:
        # loading datasets
        train_dataset = MNISTDataset(data_file='../data_mnist/mnist/mnist_train_6000_data.hkl',
                                     source_file='../data_mnist/mnist/mnist_train_6000_sources.hkl',
                                     tdim_use=opt.tdim_use,
                                     transform=None)
    
        valid_dataset = MNISTDataset(data_file='../data_mnist/mnist/mnist_valid_2000_data.hkl',
                                     source_file='../data_mnist/mnist/mnist_valid_2000_sources.hkl',
                                     tdim_use=opt.tdim_use,
                                     transform=None)
        #import pdb;pdb.set_trace()
        
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=1,
                                                   shuffle=True)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=1,
                                                   shuffle=False)

        if opt.model_name == 'clstm':
            # convolutional lstm
            from convolution_lstm_mod import *
            convlstm = CLSTM_EP(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
        elif opt.model_name == 'clstm_skip':
            # convolutional lstm with skip connection
            from convolution_lstm_mod import *
            convlstm = CLSTM_EP3(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
        elif opt.model_name == 'clstm_multi':
            # convolutional lstm with multiple layers
            from convolution_lstm_multi import *
            convlstm = CLSTM_EP_MUL(input_channels=1, hidden_channels=opt.hidden_channels,
                                kernel_size=opt.kernel_size).cuda()
    
        if opt.transfer_path != 'None':
            # Use pretrained weights for transfer learning
            print('loading pretrained model:',opt.transfer_path)
            convlstm = torch.load(opt.transfer_path)

        modelinfo.write('Model Structure \n')
        modelinfo.write(str(convlstm))
        count_parameters(convlstm,modelinfo)
        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif opt.loss_function == 'BCE': # binary cross entropy
            loss_fn = torch.nn.BCELoss()
        elif opt.loss_function == 'WeightedMSE':
            loss_fn = weighted_MSE_loss
        elif opt.loss_function == 'MaxMSE':
            loss_fn = max_MSE_loss(opt.loss_weights)
        elif opt.loss_function == 'MultiMSE':
            loss_fn = multi_MSE_loss(opt.loss_weights)

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(convlstm.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(convlstm.parameters(), lr=opt.learning_rate)
            
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
            
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor','MaxMSE','FSS_mean','lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor','MaxMSE','FSS_mean','lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss', 'RMSE', 'CSI', 'FAR', 'POD', 'Cor','MaxMSE','FSS_mean'])
    
        # training 
        for epoch in range(opt.n_epochs):
            # step scheduler
            scheduler.step()
            # training & validation
            train_epoch(epoch,opt.n_epochs,train_loader,convlstm,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt,scl)
            valid_epoch(epoch,opt.n_epochs,valid_loader,convlstm,loss_fn,
                        valid_logger,opt,scl)

        # save the trained model
        # (1) as binary 
        torch.save(convlstm,os.path.join(opt.result_path, 'trained_CLSTM.model'))
        # (2) as state dictionary
        torch.save(convlstm.state_dict(),
                   os.path.join(opt.result_path, 'trained_CLSTM.dict'))

    # test datasets if specified
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_CLSTM.model')
            print('loading pretrained model:',model_fname)
            convlstm = torch.load(model_fname)
            loss_fn = torch.nn.MSELoss()
            
        # prepare loader
        test_dataset = MNISTDataset(data_file='../data_mnist/mnist/mnist_test_2000_data.hkl',
                                     source_file='../data_mnist/mnist/mnist_test_2000_sources.hkl',
                                     tdim_use=opt.tdim_use,
                                     transform=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=False)
        
        # testing for the trained model
        for threshold in opt.eval_threshold:
            test_CLSTM_EP_mnist(test_loader,convlstm,loss_fn,opt,scl)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)


