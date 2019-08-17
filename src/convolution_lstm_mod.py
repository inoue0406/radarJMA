# Taken from the following
# https://github.com/automan000/Convolution_LSTM_PyTorch

# Modified to follow encoder-predictor structure

import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class CLSTM_EP(nn.Module):
    # Encoder-Predictor using Convolutional LSTM Cell
    def __init__(self, input_channels, hidden_channels, kernel_size):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(CLSTM_EP, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self._all_layers = []
        # initialize encoder/predictor cell
        cell_e = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.encoder = cell_e
        self._all_layers.append(cell_e)
        cell_p = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.predictor = cell_p
        self._all_layers.append(cell_p)
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        self.lastconv = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # initialize internal state
        (he, ce) = self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # encoding
        for it in range(tsize):
            # forward
            (he, ce) = self.encoder(x[:,it,:,:,:], he, ce)
        # copy internal state to predictor
        hp = he
        cp = ce
        # predictor
        xzero = Variable(torch.zeros(bsize, channels, height, width)).cuda() # ! should I put zero here?
        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        for it in range(tsize):
            (hp, cp) = self.predictor(xzero, hp, cp)
            xout[:,it,:,:,:] = self.lastconv(hp)
        return xout

class CLSTM_EP2(nn.Module):
    # A Variant of Encoder-predictor model
    # Predictor's output feeds in as a next input of LSTM cell
    def __init__(self, input_channels, hidden_channels, kernel_size):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(CLSTM_EP2, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self._all_layers = []
        # initialize encoder/predictor cell
        cell_e = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.encoder = cell_e
        self._all_layers.append(cell_e)
        cell_p = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.predictor = cell_p
        self._all_layers.append(cell_p)
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        self.lastconv = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # initialize internal state
        (he, ce) = self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # encoding
        for it in range(tsize):
            # forward
            (he, ce) = self.encoder(x[:,it,:,:,:], he, ce)
        # copy internal state to predictor
        hp = he
        cp = ce
        # predictor
        xout_prev = x[:,(tsize-1),:,:,:] # initialize with the latest var
        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        for it in range(tsize):
            (hp, cp) = self.predictor(xout_prev, hp, cp) # input previous timestep's xout
            xout[:,it,:,:,:] = self.lastconv(hp)
            xout_prev = xout[:,it,:,:,:].clone()
        return xout
    
class CLSTM_EP3(nn.Module):
    # A Variant of Encoder-predictor model
    # Predictor's output feeds in as a next input of LSTM cell
    # "Skip connection" for predictor
    def __init__(self, input_channels, hidden_channels, kernel_size):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(CLSTM_EP3, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self._all_layers = []
        # initialize encoder/predictor cell
        cell_e = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.encoder = cell_e
        self._all_layers.append(cell_e)
        cell_p = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size)
        self.predictor = cell_p
        self._all_layers.append(cell_p)
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        self.lastconv = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)
        
    def forward(self, input):
        x = input
        bsize, tsize, channels, height, width = x.size()
        # initialize internal state
        (he, ce) = self.encoder.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp, cp) = self.predictor.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        # encoding
        for it in range(tsize):
            # forward
            (he, ce) = self.encoder(x[:,it,:,:,:], he, ce)
        # copy internal state to predictor
        hp = he
        cp = ce
        # predictor
        xout_prev = x[:,(tsize-1),:,:,:] # initialize with the latest var
        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        for it in range(tsize):
            (hp, cp) = self.predictor(xout_prev, hp, cp) # input previous timestep's xout
            #xout[:,it,:,:,:] = self.lastconv(hp)
            # skip connection
            xout[:,it,:,:,:] = self.lastconv(hp) + xout_prev
            # tmp sigmoid when BCE
            #xout[:,it,:,:,:] = torch.sigmoid(self.lastconv(hp) + xout_prev)
            xout_prev = xout[:,it,:,:,:].clone()
        return xout

#if __name__ == '__main__':

    # gradient check
    #convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5, effective_step=[4]).cuda()
    #loss_fn = torch.nn.MSELoss()

    #input = Variable(torch.randn(1, 512, 64, 32)).cuda()
    #target = Variable(torch.randn(1, 32, 64, 32)).cuda()

    #output = convlstm(input)
    #output = output[0][0]
    #res = torch.autograd.gradcheck(loss_fn, (output, target), raise_exception=True)
    #print(res)

