from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        return grad_output
    

class IRLinear(nn.Linear):
    def __init__(self, in_channels, out_channels,bias=False):
        super(IRLinear, self).__init__(in_channels, out_channels, bias)
    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
        w = BinaryQuantize().apply(bw)
        output = F.linear(input, w, self.bias)
        return output
    
    
class IRConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        w = BinaryQuantize().apply(bw)
        output = F.conv2d(input, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
    

class IRConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(IRConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1)
        w = BinaryQuantize().apply(bw)
        output = F.conv1d(input, w, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
    
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
