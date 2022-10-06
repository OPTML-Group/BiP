'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
Borrow from : https://github.com/akamaster/pytorch_resnet_cifar10.git
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.layers import GetSubnetUnstructured


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNets(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10, k=0.01, unstructured=True):
        super(ResNets, self).__init__()
        self.in_planes = 16
        self.conv_layer = conv_layer
        self.k = k
        self.unstructured_pruning = unstructured

        self.conv1 = conv_layer(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = linear_layer(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.unstructured_pruning:
            score_list = []
            for (name, vec) in self.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
            scores = torch.cat(score_list)
            adj = GetSubnetUnstructured.apply(scores.abs(), self.k)

            pointer = 0
            for (name, vec) in self.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = attr * adj[pointer: pointer + numel].view_as(attr)
                            pointer += numel

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20s(conv_layer, linear_layer, **kwargs):
    return ResNets(conv_layer, linear_layer, BasicBlock, [3, 3, 3], **kwargs)


def resnet32s(conv_layer, linear_layer, **kwargs):
    return ResNets(conv_layer, linear_layer, BasicBlock, [5, 5, 5], **kwargs)


def resnet44s(conv_layer, linear_layer, **kwargs):
    return ResNets(conv_layer, linear_layer, BasicBlock, [7, 7, 7], **kwargs)


def resnet56s(conv_layer, linear_layer, **kwargs):
    return ResNets(conv_layer, linear_layer, BasicBlock, [9, 9, 9], **kwargs)
