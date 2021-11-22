

import torch.nn as nn

class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, use_bias=True):
        super(DepthwiseSeparableConv2D,self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=use_bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,  bias=use_bias)
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x