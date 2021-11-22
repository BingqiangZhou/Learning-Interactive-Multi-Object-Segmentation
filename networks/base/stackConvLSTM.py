

import torch
import torch.nn as nn
from .convLSTM import ConvLSTMCell

class StackConvLSTMCell(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True,
                 nums_conv_lstm_layer=1):
        super(StackConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nums_conv_lstm_layer = 1 if nums_conv_lstm_layer < 1 else nums_conv_lstm_layer

        self.conv_lstm_first = ConvLSTMCell(input_dim, hidden_dim, kernel_size=3, use_bias=use_bias)
        
        cell_list = []
        for i in range(self.nums_conv_lstm_layer - 1):
            cell_list.append(ConvLSTMCell(hidden_dim * 2, hidden_dim, kernel_size=3, use_bias=use_bias))

        self.conv_lstm_intermediate_cells = nn.ModuleList(cell_list)

    def forward(self, x, hidden_and_cell_list=None):
        '''
            input: [n, c, h, w]
            hidden_and_cell_list: [([n, c, h, w], [n, c, h, w]) * num_convlstm_layer]
        '''

        if hidden_and_cell_list is None:
            hidden_and_cell_list = self.__init_hidden_and_cell__(x)
        
        next_hidden_and_cell_list = []
        ## append (h_next, c_next) to list
        next_hidden_and_cell_list.append((self.conv_lstm_first(x, hidden_and_cell_list[0])))

        for i in range(self.nums_conv_lstm_layer - 1):
            input = torch.cat(next_hidden_and_cell_list[i], dim=1)
            ## append (h_next, c_next) to list
            next_hidden_and_cell_list.append((self.conv_lstm_intermediate_cells[i](input, hidden_and_cell_list[i+1])))
        
        output = next_hidden_and_cell_list[-1] # last (h_next, c_next)

        return output, next_hidden_and_cell_list
    
    def __init_hidden_and_cell__(self, x):
        n, c, h, w = x.shape
        device = x.device
        init_hidden_and_cell_list = []

        init_hidden_and_cell_list.append((torch.zeros(n, self.hidden_dim, h, w).to(device),
                torch.zeros(n, self.hidden_dim, h, w).to(device)))
        
        for i in range(self.nums_conv_lstm_layer):
            init_hidden_and_cell_list.append((torch.zeros(n, self.hidden_dim, h, w).to(device),
                    torch.zeros(n, self.hidden_dim, h, w).to(device)))

        return init_hidden_and_cell_list