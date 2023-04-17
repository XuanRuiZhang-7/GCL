import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import torch
import random
import numpy as np


# 定义归一化函数
def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)


# sn全连接层(这样的目的是为了更加稳定)
def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)


# sn卷积层(这样的目的是为了更加稳定)
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias),
                         eps=1e-6)


# sn编码(这样的目的是为了更加稳定)
def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)


# 归一化层？
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, in_features=148, out_features=384):
        super().__init__()
        self.in_features = in_features
        self.bn = batchnorm_2d(out_features, eps=1e-4, momentum=0.1, affine=False)

        self.gain = snlinear(in_features=in_features, out_features=out_features, bias=False)
        self.bias = snlinear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


# 自我注意机制？
class SelfAttention(nn.Module):
    def __init__(self, in_channels, is_generator):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        if is_generator:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                          stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                        stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                          stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                        stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                      stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma * attn_g


# 一个块
class GenBlock(nn.Module):
    def __init__(self):
        super(GenBlock, self).__init__()
        in_features = 148
        self.bn1 = ConditionalBatchNorm2d(in_features=in_features)
        self.bn2 = ConditionalBatchNorm2d(in_features=in_features)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = snconv2d(384, 384, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d1 = snconv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d2 = snconv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


# 生成网络 输入inputs.shape = tensor.size[(batch_size, 80)]
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear0 = snlinear(in_features=20, out_features=6144, bias=True)
        self.shared = nn.Embedding(10, 128)
        # 主要块
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (0): ModuleList
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (1): ModuleList
        self.blocks.append(nn.ModuleList([SelfAttention(in_channels=384, is_generator=True)]))  # (2): ModuleList
        self.blocks.append(nn.ModuleList([GenBlock()]))  # (3): ModuleList

        self.bn4 = batchnorm_2d(in_features=384)
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = snconv2d(384, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.tanh = nn.Tanh()

        # 关于cifar10的默认参数
        self.bottom = 4

    def forward(self, z, label=None):
        affine_list = []
        z0 = z
        zs = torch.split(z, 20, 1)
        z = zs[0]

        shared_label = self.shared(label)
        affine_list.append(shared_label)
        affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

        act = self.linear0(z)
        act = act.view(-1, 384, self.bottom, self.bottom)
        counter = 0
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, SelfAttention):
                    act = block(act)
                else:
                    act = block(act, affines[counter])
                    counter += 1

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out
