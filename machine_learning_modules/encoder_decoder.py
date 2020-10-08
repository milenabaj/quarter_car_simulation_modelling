"""
Encoder-decoder model without attention.

@author: Milena Bajic (DTU Compute)

"""

import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
ed_log = get_mogule_logger("encoder_decoder")

class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 1, device = 'cuda'):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # define LSTM layer
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)


    def forward(self, x_input):

        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''

        self.lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return self.lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):

    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size = 32, hidden_size = 64, output_size = 1, num_layers = 1, device = 'cuda'):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_input, encoder_hidden_states):

        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        x_input.to(self.device)
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        #print('Decoder forward - lstm_out: ',lstm_out.shape)
        lstm_out = lstm_out.squeeze(0) #-> [batch size, hidden dim]
        #print('Decoder forward - linear input: ',lstm_out.shape)
        prediction = self.linear(lstm_out)
        #print('Decoder forward - prediction: ',prediction.shape)
        return prediction, self.hidden



class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size  = 1, hidden_size = 92, target_len = 1000, 
                 use_teacher_forcing = True, device = 'cuda'):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_len = target_len
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device

        self.encoder = lstm_encoder(device = self.device, hidden_size = self.hidden_size)
        
        # decoder input: target sequence, features only taken as input hidden state
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = self.hidden_size, 
                                    device = self.device)


    def forward(self, input_batch, target_batch = None):

        '''
        : param input_batch:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        if target_batch is None:
            self.use_teacher_forcing = False # can't use teacher forcing if the output sequence is not given
            
        batch_size = input_batch.shape[1]

        # ======== ENCODER ======== #
        # Initialize hidden state
        self.encoder.init_hidden(batch_size)

        # Pass trough the encoder
        self.encoder_output, self.encoder_hidden = self.encoder(input_batch)


        # ====== DECODER ======= #
        self.decoder_input = torch.zeros([1, batch_size, 1]).to(self.device) # start of the output seq.
        self.decoder_hidden = self.encoder_hidden

        # To cuda
        self.decoder_hidden[0].to(self.device)
        self.decoder_hidden[1].to(self.device)

        # Initialize vector to store the decoder output
        self.outputs = torch.zeros([self.target_len,  batch_size, 1]).to(self.device)

        # Decoder output
        if self.use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for t in range(self.target_len):
                self.decoder_output, self.decoder_hidden = self.decoder(self.decoder_input, self.decoder_hidden)
                self.outputs[t] = self.decoder_output
                self.decoder_input = target_batch[t,:,:].unsqueeze(0).to(self.device) # current target will be the input in the next timestep

        else:
            # Without teacher forcing: use its own predictions as the next input
            for t in range(self.target_len):
                self.decoder_output, self.decoder_hidden = self.decoder(self.decoder_input, self.decoder_hidden)
                self.outputs[t] = self.decoder_output
                self.decoder_input = self.decoder_output.unsqueeze(0).to(self.device)

        return self.outputs

