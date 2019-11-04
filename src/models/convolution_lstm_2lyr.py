# Two layers model combining SVG model and ConvLSTM

import torch
import torch.nn as nn
from torch.autograd import Variable

# import encoder-decoder model
import models.dcgan_128 as ed_model
import models.lstm as lstm_models
from models.convolution_lstm_mod import *

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CLSTM_2lyr(nn.Module):
    # Two-layers model
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(CLSTM_2lyr, self).__init__()
        # temp
        predictor_rnn_layers = 2
        hidden_dim = 128
        rnn_size = 256
        batch_size = 20
        # Initialize encoder and decoder
        self.encoder = ed_model.encoder(hidden_dim, input_channels)#.cuda()
        self.decoder = ed_model.decoder(hidden_dim, input_channels)#.cuda()
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        # Initialize frame predictor
        self.frame_predictor = lstm_models.lstm(hidden_dim, hidden_dim, rnn_size,
                                           predictor_rnn_layers, batch_size)#.cuda()
        self.frame_predictor.apply(init_weights)
        # Initialize conv predictor
        self.convlstm = CLSTM_EP(input_channels=1, hidden_channels=hidden_channels,
                            kernel_size=kernel_size)#.cuda()
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        self.frame_predictor.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        # initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()

        # time step by convlstm
        xout = self.convlstm(x)
        #xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()

        # time step for past frames
        for it in range(0, tsize):
            x_in = x[:,it-1,:,:,:] # use ground truth frame for the first half
            h, skip = self.encoder(x_in)
            h_pred = self.frame_predictor(h)
            x_pred = self.decoder([h_pred, skip])
        # time step for future frames
        for it in range(0, tsize):
            x_in = x_pred # use predicted frame for the second half (NOT use ground truth)
            _, skip = self.encoder(x_in)
            h = h_pred
            h_pred = self.frame_predictor(h)
            x_pred = self.decoder([h_pred, skip])
            xout[:,it,:,:,:] = xout[:,it,:,:,:] + x_pred
            #import pdb;pdb.set_trace()
        return xout
    
