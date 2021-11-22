

import torch
import torch.nn as nn

class VisualAttention(torch.nn.Module):
    def __init__(self, rnn_cells, foreground_convs, background_convs, channel_of_per_interactive_map=1):
        
        super(VisualAttention, self).__init__()

        self.channel_of_per_interactive_map = channel_of_per_interactive_map

        self.rnn_cells = rnn_cells

        ## get foreground visual attention
        self.foreground_convs = foreground_convs
                                            
        ## get background visual attention
        self.background_convs = background_convs
        
        self.normalization_function = nn.Sigmoid()
        
    # def normalization_function(self, x):
    #     x = torch.relu(x)
    #     x = 1 - torch.exp(-x)
    #     return x

    def forward(self, feature_map, interactive_map):
        '''
            feature_map: [n, f, h, w]
            interactive_map: [n, c, h, w], not include background
            visual_attention_maps: [n, c+1, h, w], the first channel is background
        '''
        
        output_maps = []

        ## get foreground visual attention map
        hidden_and_cell_list = None
        for current_interactive_map in torch.split(interactive_map, self.channel_of_per_interactive_map, dim=1):
            feature_combined = torch.cat([feature_map, current_interactive_map], dim=1)
            outputs, hidden_and_cell_list = self.rnn_cells(feature_combined, hidden_and_cell_list)
            hidden_state, cell_state = outputs
            output = self.foreground_convs(hidden_state)
            output_maps.append(output)
        
        ## get background visual attention map
        hidden_state, cell_state = hidden_and_cell_list[-1]
        output = self.background_convs(hidden_state)
        output_maps.insert(0, output)
        output_maps = torch.cat(output_maps, dim=1)

        output_maps = self.normalization_function(output_maps)

        return output_maps