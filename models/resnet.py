from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from torch.nn.utils.weight_norm import WeightNorm
import cv2
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)  # split the weight update component to direction and norm

        if outdim <= 200:
            self.scale_factor = 2;  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10;  # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(
            x_normalized)  # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores


class Generator(nn.Module):
    def __init__(self, norm=False, latent_dim=200, class_dim=200, feat_dim=2048, hidden_dim=2048):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim + class_dim
        self.feat_dim = feat_dim
        self.norm = norm
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(weights_init)

    def forward(self, z, label):
        z = torch.cat((z, label), 1)
        img = self.model(z)
        if self.norm:
            img = F.normalize(img)
        return img


class Generator_IMG_128(nn.Module):
    def __init__(self,channel,n_classes=100, nz=200, ngf=64):
        super(Generator_IMG_128, self).__init__()
        nc = channel  # channel
                                    # (300,1,1)
        self.netG = nn.Sequential(nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8),    # 2     [512, 4, 4]
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),         # 4       [256, 8, 8]
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),   # [256, 16, 16]
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),     # [128, 32, 32]
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),         # 8
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                                  nn.Tanh()  # (N,nc, 128,128)
                                )


        # self.apply(weights_init)
        self.img_idx = 1

    def forward(self, z, label):
        # input z : [bs, laten_dim]   label: [bs, num_class]
        bs = z.shape[0]
        z = z.view(bs, -1, 1, 1)
        label = label.view(bs, -1, 1, 1)
        data = torch.cat((z, label), dim=1)     # (N,nz+n_classes,1,1)(128,300,1,1)
        img = self.netG(data)
        # print(img.shape)

        # #======================== not used =============================================#
        # for i in range(img.shape[0]):
        #     cv_img = img[0].cpu().detach().numpy()
        #     #mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010)
        #     cv_img[0,:,:] = (cv_img[0,:,:] * 0.2023 + 0.4914) * 255
        #     cv_img[1, :, :] = (cv_img[1, :, :] * 0.1994 + 0.4822) * 255
        #     cv_img[2, :, :] = (cv_img[2, :, :] * 0.2010 + 0.4465) * 255
        #     cv_img = np.clip(cv_img, 0, 255)
        #     cv_img = cv_img.transpose(1,2,0)
        #     cv2.imwrite("../checkpoints/img_save/{}.jpg".format(self.img_idx), cv_img)
        #     self.img_idx += 1
        # # =====================================================================#
        return img


class Generator_IMG(nn.Module):
    def __init__(self, channel,n_classes=200, nz=100, ngf=64):
        super(Generator_IMG, self).__init__()
        nc = channel

        self.netG = nn.Sequential(nn.ConvTranspose2d(nz + n_classes, ngf * 8, 4, 1, 0, bias=False),nn.BatchNorm2d(ngf * 8),  # 2
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),# 4
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),# 8
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                                  nn.Tanh()  # (N,nc, 128,128)
                                  )
        # self.apply(weights_init)
        self.img_idx = 1

    def forward(self, z, label):
        # input z : [bs, laten_dim]   label: [bs, num_class]
        bs = z.shape[0]
        z = z.view(bs, -1, 1, 1)
        label = label.view(bs, -1, 1, 1)
        data = torch.cat((z, label), dim=1)  # (N,nz+n_classes,1,1)
        img = self.netG(data)
        return img


class Discriminator_IMG(nn.Module):
    def __init__(self, channel, n_classes=100, ndf=64):
        super(Discriminator_IMG, self).__init__()
        # self.apply(weights_init)
        # =================================#
        nc = channel
        self.netD = nn.Sequential(nn.Conv2d(nc + n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),  # 512,16,16
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                                    nn.Flatten(),  # (N,1)
                                    nn.Sigmoid()
                                  )

    def forward(self, img, label):
        # input img [bs, 3, 32, 32]          label [bs, num_class]
        bs = img.shape[0]
        label = label.view(bs, -1, 1, 1)
        label = label.repeat(1, 1, img.size(2), img.size(3))   # 加到数据上
        data = torch.cat((img, label), dim=1)    # (N,nc + n_classes, 32, 32)
        score = self.netD(data)
        return score


class Discriminator(nn.Module):
    def __init__(self, feat_dim=2048, class_dim=200, hidden_dim=2048, condition='projection'):
        super(Discriminator, self).__init__()

        self.feat_dim = feat_dim
        self.model = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.valid = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)
        self.classifer = nn.Linear(hidden_dim, class_dim)
        self.projection = nn.Linear(class_dim, hidden_dim, bias = False)
        self.condition = condition

    def forward(self, img, label):
        hidden = self.model(img)
        var = self.projection(label)
        validity = (var * hidden).sum(dim=1).reshape(-1, 1) + self.valid(hidden)
        classifier = self.classifer(hidden)
        return validity, classifier

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, norm=False, Embed=True, feat_dim=2048, embed_dim=2048):
        super(ResNet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.norm = norm
        self.Embed = Embed
        self.relu = nn.ReLU(inplace=True)

        self.embed = nn.Linear(feat_dim, embed_dim)
        
    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        feat = x.view(x.size(0), -1)

        # if self.Embed==True:
        #    x=self.embed(feat)
        # x=self.relu(x)

        if self.norm:
            x = F.normalize(x)
        return feat


class ModelCNN(nn.Module):

    def __init__(self, class_dim=10, norm=False, method='softmax'):
        super(ModelCNN, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 16)
        if method == 'softmax' or method == 'BCE':
            self.embed = nn.Linear(16, class_dim)
        elif method == 'dist':
            self.embed = distLinear(16, class_dim)

    def forward(self, input_):
        h1 = F.relu(F.max_pool2d(self.conv1(input_), 2))
        h2 = F.relu(F.max_pool2d(self.conv2(h1), 2))
        h2 = h2.view(-1, 512)
        h3 = F.relu(self.fc1(h2))
        h4 = F.relu(self.fc2(h3))
        if self.norm:
            h4 = F.normalize(h4)
        return h4


class ClassifierMLP(nn.Module):

    def __init__(self, class_dim=10, norm=False, method='softmax'):
        super(ClassifierMLP, self).__init__()
        self.norm = norm
        self.fc0 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 16)
        if method == 'softmax' or method == 'BCE':
            self.embed = nn.Linear(16, class_dim)
        elif method == 'dist':
            self.embed = distLinear(16, class_dim)

    def forward(self, input_):
        input_ = input_.view(input_.size(0), -1)
        h1 = F.relu(self.fc0(input_))
        h2 = F.relu(self.fc1(h1))

        h3 = F.relu(self.fc2(h2))
        h4 = F.relu(self.fc3(h3))
        if self.norm:
            h4 = F.normalize(h4)
        return h4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_ImageNet(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False, norm=False, Embed=True, feat_dim=512, embed_dim=512):
        super(ResNet_ImageNet, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norm = norm
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.embed = nn.Linear(feat_dim, embed_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out_features = out.view(out.size(0), -1)
        if self.norm:
            out_features = F.normalize(out_features)
        return out_features


# cifar
class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False, norm=False, Embed=True, feat_dim=2048, embed_dim=2048):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norm = norm
        self.relu = nn.ReLU(inplace=True)

        self.embed = nn.Linear(feat_dim, embed_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)     # [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out_features = out.view(out.size(0), -1)
        if self.norm:
            out_features = F.normalize(out_features)
        return out_features


class ResNet_Cifar10(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False, norm=False, Embed=True, feat_dim=2048, embed_dim=2048):
        super(ResNet_Cifar10, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norm = norm
        self.relu = nn.ReLU(inplace=True)

        self.embed = nn.Linear(512, embed_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)     # [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out_features = out.view(out.size(0), -1)
        if self.norm:
            out_features = F.normalize(out_features)
        return out_features

# mnist-32*32
class ResNet_Mnist_32(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False, norm=False, Embed=True, feat_dim=2048, embed_dim=10):
        super(ResNet_Mnist_32, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norm = norm
        self.relu = nn.ReLU(inplace=True)

        self.embed = nn.Linear(feat_dim, embed_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)     # [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out_features = out.view(out.size(0), -1)
        if self.norm:
            out_features = F.normalize(out_features)
        return out_features


# 128*128
class ResNet_Mnist(nn.Module):
    def __init__(self, block, num_blocks, pretrained=False, norm=False, Embed=True, feat_dim=2048, embed_dim=10):
        super(ResNet_Mnist, self).__init__()
        self.in_planes = 64

        self.layer0_conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer0_bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.norm = norm
        self.relu = nn.ReLU(inplace=True)

        # self.embed = nn.Linear(feat_dim, embed_dim)
        self.embed = nn.Linear(8192, embed_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)     # [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.layer0_bn1(self.layer0_conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out_features = out.view(out.size(0), -1)
        if self.norm:
            out_features = F.normalize(out_features)
        return out_features


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet18_imagenet(**kwargs):
    return ResNet_ImageNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_cifar(**kwargs):
    return ResNet_Cifar(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_mnist_32(**kwargs):
    return ResNet_Mnist_32(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet18_mnist(**kwargs):
    return ResNet_Mnist(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)
