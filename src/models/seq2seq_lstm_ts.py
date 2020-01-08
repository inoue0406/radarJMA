# Time series seq2seq model

import random

import torch
import torch.nn as nn
from torch.autograd import Variable

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
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        # initialize weights
        for name, param in encoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        for name, param in decoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
        
    def forward(self, in_seq, trg_seq, teacher_forcing_ratio = 0.5):
            
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
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(in_seq)

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
        
        
