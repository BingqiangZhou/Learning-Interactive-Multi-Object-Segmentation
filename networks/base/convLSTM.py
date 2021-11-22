
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, use_bias=True):

        super(ConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim + self.hidden_dim,
                        out_channels=self.hidden_dim * 4,
                        kernel_size=kernel_size, padding=padding, bias=use_bias),
            # nn.ReLU(inplace=True),
            # nn.GroupNorm(4, self.hidden_dim * 4), # group=4
            # nn.ReLU(inplace=True),
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, input_tensor):
        device = input_tensor.device # self.conv.weight.device
        b, _, h, w = input_tensor.shape
        return (torch.zeros(b, self.hidden_dim, h, w, device=device),
                torch.zeros(b, self.hidden_dim, h, w, device=device))