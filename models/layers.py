import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GetSubnetFilter(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, p=1):  # binarization
        # Get the subnetwork by sorting the scores and using the top k%
        score_L1_norm = torch.norm(scores.flatten(start_dim=1, end_dim=-1), p=p, dim=1)
        _, idx = score_L1_norm.sort()
        j = int((1 - k) * scores.shape[0])

        # flat_out and out access the same memory.
        out = scores.clone()
        flat_out = out.flatten(start_dim=1, end_dim=-1)  # share the same memory
        flat_out[idx[:j], :] = 0
        flat_out[idx[j:], :] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class GetSubnetChannel(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, p=1):  # binarization
        # Get the subnetwork by sorting the scores and using the top k%
        score_L1_norm = torch.norm(torch.norm(scores, p=p, dim=[2, 3]), p=p, dim=0)
        _, idx = score_L1_norm.sort()
        j = int((1 - k) * scores.shape[1])

        # flat_out and out access the same memory.
        out = scores.clone()
        out[:, idx[:j], :, :] = 0
        out[:, idx[j:], :, :] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None


class GetSubnetUnstructured(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class SubnetConvFilter(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConvFilter, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.p = 1

    def set_prune_rate(self, k):
        self.k = k
        self.p = 2

    def forward(self, x):
        adj = GetSubnetFilter.apply(self.popup_scores.abs(), self.k, self.p)
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetConvChannel(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConvChannel, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0
        self.p = 1

    def set_prune_rate(self, k):
        self.k = k
        self.p = 2

    def forward(self, x):
        adj = GetSubnetChannel.apply(self.popup_scores.abs(), self.k, self.p)

        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetConvUnstructured(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
    ):
        super(SubnetConvUnstructured, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.w = 0

    def set_prune_rate(self, k):
        self.k = k

    def forward(self, x):
        x = F.linear(x, self.w, self.bias)
        return x
