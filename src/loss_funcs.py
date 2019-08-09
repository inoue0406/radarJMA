# custom loss function for Pytorch

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

def weighted_MSE_loss(output, target):
    # custom loss function weighing heavy rains
    # First introduced by Shi et. al. (2017)
    mask_02 = (target<2.0).float()
    mask_05 = ((target<5.0) * (target>=2.0)).float()
    mask_10 = ((target<10.0) * (target>=5.0)).float()
    mask_30 = ((target<30.0) * (target>=10.0)).float()
    mask_XX = (target>=30.0).float()
    loss = 1.0 * torch.mean((mask_02*(output - target))**2)+ \
           2.0 * torch.mean((mask_05*(output - target))**2)+ \
           5.0 * torch.mean((mask_10*(output - target))**2)+ \
           10.0 * torch.mean((mask_30*(output - target))**2)+ \
           30.0 * torch.mean((mask_XX*(output - target))**2)
    return loss

class max_MSE_loss(nn.Module):
    # custom loss function weighing heavy rains   
    def __init__(self, weights):
        super(max_MSE_loss, self).__init__()
        self.weights = weights
        assert len(weights) == 2, 'size of weights should be {0}, but given {1}'.format(2,len(weights))

    def forward(self, output, target):
        mse = torch.mean((output - target)**2)
        # take max using MaxPool (in order to be differentiable)
        out2 = output.view(output.shape[0:2] + output.shape[3:5])
        omax = F.max_pool2d(out2, kernel_size=out2.size()[2:])
        tgt2 = target.view(target.shape[0:2] + target.shape[3:5])
        tmax = F.max_pool2d(tgt2, kernel_size=tgt2.size()[2:])
        maxmse = torch.mean((omax - tmax)**2)
        loss = self.weights[0] * mse + self.weights[1] * maxmse
        return loss

class multi_MSE_loss(nn.Module):
    # multi scale MSE loss
    def __init__(self, weights):
        super(multi_MSE_loss, self).__init__()
        self.weights = weights
        assert len(weights) == 4, 'size of weights should be {0}, but given {1}'.format(2,len(weights))

    def forward(self, output, target):
        mse = torch.mean((output - target)**2)
        # take max using MaxPool (in order to be differentiable)
        out2 = output.view(output.shape[0:2] + output.shape[3:5])
        omax2 = F.max_pool2d(out2, kernel_size=out2.size()[2:])
        tgt2 = target.view(target.shape[0:2] + target.shape[3:5])
        tmax2 = F.max_pool2d(tgt2, kernel_size=tgt2.size()[2:])
        maxmse = torch.mean((omax2 - tmax2)**2)
        # maxpooing for 20x20 grid
        out3 = output.view(output.shape[0:2] + output.shape[3:5])
        omax3 = F.max_pool2d(out3, kernel_size=20, stride=20)
        tgt3 = target.view(target.shape[0:2] + target.shape[3:5])
        tmax3 = F.max_pool2d(tgt3, kernel_size=20, stride=20)
        mse_20 = torch.mean((omax3 - tmax3)**2)
        # maxpooing for 5x5 grid
        out4 = output.view(output.shape[0:2] + output.shape[3:5])
        omax4 = F.max_pool2d(out4, kernel_size=5, stride=5)
        tgt4 = target.view(target.shape[0:2] + target.shape[3:5])
        tmax4 = F.max_pool2d(tgt4, kernel_size=5, stride=5)
        mse_5 = torch.mean((omax4 - tmax4)**2)
        #import pdb; pdb.set_trace()
        loss = self.weights[0]*mse + self.weights[1]*mse_5 + self.weights[2]*mse_20 + self.weights[3]*maxmse
        return loss
