
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first= False, num_layers=1, bidirectional = False, dropout= 0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)


    def reset_params(self):
        