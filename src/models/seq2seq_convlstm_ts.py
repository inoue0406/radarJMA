# Time series seq2seq model

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout,
                           batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, in_seq):
            
        #in_seq = [in_seq len, batch size]
        
        #in_seq = self.dropout(in_seq)
        
        outputs, (hidden, cell) = self.rnn(in_seq)
        
        #outputs = [in_seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout = dropout,
                           batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(1) #insert time dim in the middle
        
        #input = [1, batch size]
        
        #input = self.dropout(input)

        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

# use VGG type convolution layer as 2d info encoder
class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class conv_encoder_layer(nn.Module):
    def __init__(self, dim, nc=1):
        super(conv_encoder_layer, self).__init__()
        self.dim = dim
        # 128 x 128
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        # 64 x 64
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 32 x 32 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 16 x 16
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 8 x 8
        self.c5 = nn.Sequential(
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 128 -> 64
        h2 = self.c2(self.mp(h1)) # 64 -> 32
        h3 = self.c3(self.mp(h2)) # 32 -> 16
        h4 = self.c4(self.mp(h3)) # 16 -> 8
        h5 = self.c5(self.mp(h4)) # 8 -> 4
        h6 = self.c6(self.mp(h5)) # 4 -> 1
        #return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]
        return self.mp(h6).view(-1, self.dim) # added for 200x200 size
    
class Seq2SeqConv(nn.Module):
    def __init__(self, encoder, decoder, conv_hidden_dim, device):
        super().__init__()

        self.hidden_dim = conv_hidden_dim
        self.input_channels = 1
        self.conv_encoder = conv_encoder_layer(self.hidden_dim,self.input_channels)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        # initialize weights
        self.conv_encoder.apply(init_weights)
        for name, param in self.encoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        for name, param in self.decoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    def forward(self, in_2d, in_seq, trg_seq, teacher_forcing_ratio = 0.5):

        #in_2d : 2d input
        #in_seq = [batch size, in_seq len, num of vars]   * input sequence
        #trg_seq = [batch size, trg_seq len, num of vars] * target sequence
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg_seq.shape[0]
        trg_len = trg_seq.shape[1]
        trg_nvar= trg_seq.shape[2]
        trg_vocab_size = self.decoder.output_dim
    
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_nvar).to(self.device)
        summarized = torch.zeros(batch_size, trg_len, self.hidden_dim).to(self.device)

        # apply conv_encoder layer to summarize 2d variable into 1d vector
        for t in range(0, trg_len):
            summarized[:,t,:] = self.conv_encoder(in_2d[:,t,:,:,:])
        
        X = torch.cat((in_seq,summarized),2)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(X)

        # use last "rmax_100" as initial value
        input = in_seq[:,trg_len-1,0]
        input = input.unsqueeze(1)
        
        for t in range(0, trg_len):
            #insert previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            #place predictions in a tensor
            outputs[:,t,:] = output[:,0,:]
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next value as next input
            #if not, use predicted value
            input = trg_seq[:,t,:] if teacher_force else output[:,0,:]
            
        return outputs
        
        
