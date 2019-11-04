# Probabilistic rainfall forecast by
# Variational Convolutional LSTM

# ConvLSTMCell Taken from the following
# https://github.com/automan000/Convolution_LSTM_PyTorch

# Modified to follow encoder-predictor structure,
# and to have  multi-layer structure

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
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
        self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class ConvLayers(nn.Module):
    # A block of convolution layers for downsizing
    def __init__(self, input_channels, output_channels, mid_chs, kernel_size, bias=True):
        super(ConvLayers, self).__init__()
        print('ConvLayers----------------')
        print('input channels:',input_channels)
        print('output channels:',output_channels)
        print('mid channels:',mid_chs)
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=mid_chs[0],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn1 = nn.BatchNorm2d(mid_chs[0])
        self.conv2 = nn.Conv2d(in_channels=mid_chs[0], out_channels=mid_chs[1],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn2 = nn.BatchNorm2d(mid_chs[1])
        self.conv3 = nn.Conv2d(in_channels=mid_chs[1], out_channels=mid_chs[2],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn3 = nn.BatchNorm2d(mid_chs[2])
        self.conv4 = nn.Conv2d(in_channels=mid_chs[2], out_channels=mid_chs[3],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn4 = nn.BatchNorm2d(mid_chs[3])
        self.conv5 = nn.Conv2d(in_channels=mid_chs[3], out_channels=output_channels,
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn5 = nn.BatchNorm2d(output_channels)

        # Define the leaky relu activation function
        self.l_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        #print("in ConvLayers input shape:",x.shape)
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.l_relu(conv1)
        #print("in ConvLayers output shape:",conv1.shape)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.l_relu(conv2)
        #print("in ConvLayers output shape:",conv2.shape)
        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.l_relu(conv3)
        #print("in ConvLayers output shape:",conv3.shape)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.l_relu(conv4)
        #print("in ConvLayers output shape:",conv4.shape)
        conv5 = self.conv5(conv4)
        conv5 = self.bn5(conv5)
        conv5 = self.l_relu(conv5)
        #print("in ConvLayers output shape:",conv5.shape)
        return conv5

class DeconvLayers(nn.Module):
    # A block of convolution layers for downsizing
    def __init__(self, input_channels, output_channels, mid_chs, kernel_size, bias=True):
        super(DeconvLayers, self).__init__()
        print('DeconvLayers----------------')
        print('input channels:',input_channels)
        print('output channels:',output_channels)
        print('mid channels:',mid_chs)

        self.deconv5 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=mid_chs[3],
                                        kernel_size=kernel_size, stride=2, padding=1, bias=bias)
        self.dbn5 = nn.BatchNorm2d(mid_chs[2])
        self.deconv4 = nn.ConvTranspose2d(in_channels=mid_chs[3], out_channels=mid_chs[2],
                                        kernel_size=kernel_size, stride=2, padding=1, bias=bias)
        self.dbn4 = nn.BatchNorm2d(mid_chs[2])
        self.deconv3  = nn.ConvTranspose2d(in_channels=mid_chs[2],  out_channels=mid_chs[1],
                                           kernel_size=2, stride=2, padding=0, bias=bias)
        self.dbn3 = nn.BatchNorm2d(mid_chs[1])
        self.deconv2 = nn.ConvTranspose2d(in_channels=mid_chs[1], out_channels=mid_chs[0],
                                          kernel_size=2, stride=2, padding=0, bias=bias)
        self.dbn2 = nn.BatchNorm2d(mid_chs[0])
        self.deconv1 = nn.ConvTranspose2d(in_channels=mid_chs[0], out_channels=output_channels,
                                          kernel_size=2, stride=2, padding=0, bias=bias)
        
        # Define the leaky relu activation function
        self.l_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        #print("in DeconvLayers input shape:",x.shape)

        z = self.deconv5(x)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv4(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv3(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        #
        z = self.deconv2(z)
        z = self.l_relu(z)
        ## Add skip connections
        #z = torch.cat([z, self.skip_values['conv3']])
        output = self.deconv1(z)
        #print("in DeconvLayers output shape:",output.shape)
        return output

class Var_Enc(nn.Module):
    # Variational encoder structure for encoding sequence
    # * Unlike ordinary VAE, this netowrk does not have "decoding" layers
    def __init__(self, input_channels, output_channels, mid_chs, kernel_size, bias=True):
        super(Var_Enc, self).__init__()
        print('Variational Encoder----------------')
        print('input channels:',input_channels)
        print('output channels:',output_channels)
        print('mid channels:',mid_chs)

        # Encoder Architecture
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=mid_chs[0],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn1 = nn.BatchNorm2d(mid_chs[0])
        self.conv2 = nn.Conv2d(in_channels=mid_chs[0], out_channels=mid_chs[1],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn2 = nn.BatchNorm2d(mid_chs[1])
        self.conv3 = nn.Conv2d(in_channels=mid_chs[1], out_channels=mid_chs[2],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn3 = nn.BatchNorm2d(mid_chs[2])
        self.conv4 = nn.Conv2d(in_channels=mid_chs[2], out_channels=mid_chs[3],
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn4 = nn.BatchNorm2d(mid_chs[3])
        # Convolution for mu and sigma
        self.conv5_mu = nn.Conv2d(in_channels=mid_chs[3], out_channels=output_channels,
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn5_mu = nn.BatchNorm2d(output_channels)
        self.conv5_sig = nn.Conv2d(in_channels=mid_chs[3], out_channels=output_channels,
                               kernel_size=kernel_size, padding=1, stride=2, bias=bias)
        self.bn5_sig = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

        # Define the leaky relu activation function
        self.l_relu = nn.LeakyReLU(0.1)

    def encode(self, x):
        # Encoding the input image to the mean and var of the latent distribution
        bs, _, _, _ = x.shape
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.l_relu(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.l_relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.bn3(conv3)
        conv3 = self.l_relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = self.bn4(conv4)
        conv4 = self.l_relu(conv4)

        # calc mu and sigma
        conv5_mu = self.conv5_mu(conv4)
        conv5_mu = self.bn5_mu(conv5_mu)
        conv5_mu = self.l_relu(conv5_mu)
        #
        conv5_sig = self.conv5_sig(conv4)
        conv5_sig = self.bn5_sig(conv5_sig)
        conv5_sig = self.l_relu(conv5_sig)

        # no flatten
        mu = conv5_mu
        logvar = conv5_sig
        ## flatten
        #mu = conv5_mu.view((bs, -1))
        #logvar = conv5_sig.view((bs, -1))
                
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAR_CLSTM_EP(nn.Module):
    # Variational Encoder-Predictor using Convolutional LSTM Cell
    # Multiple layers version
    def __init__(self, input_channels, hidden_channels, mid_channels, tdim_use, kernel_size, bias=True):
        # input_channels (scalar) 
        # hidden_channels (scalar) 
        super(VAR_CLSTM_EP, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self._all_layers = []
        # initialize encoder/predictor cell
        # LSTM1
        cell_e1 = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.bias)
        self.encoder1 = cell_e1
        self._all_layers.append(cell_e1)
        cell_p1 = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.bias)
        self.predictor1 = cell_p1
        self._all_layers.append(cell_p1)
        # LSTM2 (small)
        cell_e2 = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.bias)
        self.encoder2 = cell_e2
        self._all_layers.append(cell_e2)
        cell_p2 = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.bias)
        self.predictor2 = cell_p2
        self._all_layers.append(cell_p2)
        
        # Conv&Deconv layers to work with LSTM
        conv_lyr = ConvLayers(self.hidden_channels, self.hidden_channels, self.mid_channels, self.kernel_size, self.bias)
        deconv_lyr = DeconvLayers(self.hidden_channels, self.hidden_channels, self.mid_channels, self.kernel_size, self.bias)
        self.conv_lyr = conv_lyr
        self.deconv_lyr = deconv_lyr
        self._all_layers.append(conv_lyr)
        self._all_layers.append(deconv_lyr)
        
        # Variational encoding
        var_enc = Var_Enc(tdim_use*2, 1, self.mid_channels, self.kernel_size, self.bias)
        self.var_enc = var_enc
        self._all_layers.append(var_enc)
        
        # last conv layer for prediction
        self.padding = int((self.kernel_size - 1) / 2)
        self.lastconv = nn.Conv2d(self.hidden_channels, self.input_channels, self.kernel_size, 1, self.padding, bias=True)
        
    def forward(self, x_vae, x):
        bsize, tsize, channels, height, width = x.size()
        # zero matrix for original and small size
        xzero = Variable(torch.zeros(bsize, channels, height, width)).cuda()
        h_small = round(height / (2**5) + 0.5)
        w_small = round(width / (2**5) + 0.5)
        xz_small = Variable(torch.zeros(bsize, channels, h_small, w_small)).cuda()
        # Encoding
        if self.training:
            # When in training mode, variational encoder to generate z
            z,mu,logvar = self.var_enc(x_vae)
        else:
            # When in testing mode, sample z from prior (normal)
            mu = Variable(torch.zeros(bsize, channels, h_small, w_small)).cuda()
            # when var is 1, logvar should be 0
            logvar = Variable(torch.zeros(bsize, channels, h_small, w_small)).cuda()
            z = torch.randn_like(mu)
        # initialize internal state
        (he1, ce1) = self.encoder1.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (hp1, cp1) = self.predictor1.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(height, width))
        (he2, ce2) = self.encoder2.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(h_small, w_small))
        (hp2, cp2) = self.predictor2.init_hidden(batch_size=bsize, hidden=self.hidden_channels, shape=(h_small, w_small))
        # encoding
        for it in range(tsize):
            # convolution for downsizing
            (he2) = self.conv_lyr(he1) + z
            (ce2) = self.conv_lyr(ce1) + z
            # forward
            (he1, ce1) = self.encoder1(x[:,it,:,:,:], he1, ce1)
            (he2, ce2) = self.encoder2(xz_small, he2, ce2)
            # deconvolution for upsizing
            (he1) = he1 + self.deconv_lyr(he2)
            (ce1) = ce1 + self.deconv_lyr(ce2)
        # copy internal state to predictor1
        hp1 = he1
        cp1 = ce1
        hp2 = he2
        cp2 = ce2
        # predictor
        xout = Variable(torch.zeros(bsize, tsize, channels, height, width)).cuda()
        for it in range(tsize):
            # convolution for downsizing
            (hp2) = self.conv_lyr(hp1)
            (cp2) = self.conv_lyr(cp1)
            # forward
            (hp1, cp1) = self.predictor1(xzero, hp1, cp1)
            (hp2, cp2) = self.predictor2(xz_small, hp2, cp2)
            # deconvolution for upsizing
            (hp1) = hp1 + self.deconv_lyr(hp2)
            (cp1) = cp1 + self.deconv_lyr(cp2)
            # last convolution
            xout[:,it,:,:,:] = torch.sigmoid(self.lastconv(hp1))
        return xout,mu,logvar

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

