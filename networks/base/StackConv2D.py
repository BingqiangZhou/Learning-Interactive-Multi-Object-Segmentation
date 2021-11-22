

import torch.nn as nn
from functools import partial

from .depthwiseSeparableConv2D import DepthwiseSeparableConv2D

class StackConv2D(nn.Module):
    def __init__(self, conv_type='DSConv', norm_type='BN', activation_type='ReLU', in_channels=[], out_channel=128,
                kernel_size=3, stride=1, padding=1, dilation=1, groups=1, use_bias=True, 
                nums_norm_channels_per_groups=32, affine=True, extra_1x1_conv_for_last_layer=False):
        super(StackConv2D, self).__init__()

        '''
            the followed parameters for conv layer:
                'conv_type', 'in_channels', 'out_channel', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'use_bias'
            
            the followed parameters for norm layer:
                'norm_type', 'nums_norm_channels_per_groups', 'affine'
            
        '''
        
        support_conv_list = ['DSConv', 'Conv']
        assert conv_type in support_conv_list, \
            f'not support this conv type [{conv_type}]'
        
        support_norm_list =  ['BN', 'GN', 'LN', 'IN']
        assert norm_type in support_norm_list, \
            f'not support this conv type [{norm_type}]'

        if conv_type == 'DSConv':
            conv2D = partial(DepthwiseSeparableConv2D, 
                            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, use_bias=use_bias)
        else: # conv_type == 'Conv'
            conv2D = partial(nn.Conv2d, 
                            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=use_bias)

        if norm_type == 'BN':
            norm_layer = partial(nn.BatchNorm2d, track_running_stats=False) # nhw
        elif norm_type == 'GN':
            norm_layer = lambda in_channels, affine: nn.GroupNorm(in_channels//nums_norm_channels_per_groups, in_channels, affine=affine) # (c/g)hw
        elif norm_type == 'LN':
            norm_layer = lambda in_channels, affine: nn.GroupNorm(1, in_channels, affine=affine)# chw
        else: # norm_type == 'IN':
            norm_layer = nn.InstanceNorm2d
            # norm_layer = lambda in_channels, affine: nn.GroupNorm(in_channels, in_channels, affine=affine) #hw

        ## Current only support non-parameters activation layer
        activation_layer = getattr(nn, activation_type)

        conv_list = []
        for i in range(len(in_channels) - 1):
            conv_list.append(nn.Sequential(
                conv2D(in_channels[i], in_channels[i+1]),
                norm_layer(in_channels[i+1], affine=affine),
                activation_layer(inplace=True),
            ))
        
        ## last layer
        if extra_1x1_conv_for_last_layer:
            conv_list.append(conv2D(in_channels[-1], in_channels[-1]))
            conv_list.append(nn.Conv2d(in_channels[-1], out_channel, kernel_size=1))
        else:
            conv_list.append(conv2D(in_channels[-1], out_channel))

        self.stack_conv_module = nn.ModuleList(conv_list)
    
    def forward(self, x):
        out = x
        for module in self.stack_conv_module:
            out = module(out)
        return out

        



