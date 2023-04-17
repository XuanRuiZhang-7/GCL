from __future__ import absolute_import, print_function
import argparse
import getpass
import os
import sys
import torch.utils.data
import pdb
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
import models
import matplotlib.pyplot as plt

from utils import RandomIdentitySampler, mkdir_if_missing, logging, display, truncated_z_sample
from torch.optim.lr_scheduler import StepLR
import numpy as np
from ImageFolder import *
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from evaluations import extract_features, pairwise_distance
from models.resnet import Generator, Discriminator, ClassifierMLP, ModelCNN, Generator_IMG, Discriminator_IMG
import torch.autograd as autograd
import scipy.io as sio
from CIFAR100 import CIFAR100
from MNIST import MNIST
from tqdm import tqdm


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

cudnn.benchmark = True
from copy import deepcopy
import torchvision
num_classes=10
np.random.seed(33)
class_index = [5,8]
traindir = 'D:/Dataset/mnist'
BatchSize=20
transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

target_transform = np.random.permutation(num_classes)
trainset = MNIST(root=traindir, train=True, download=False, transform=transform_train,
                 target_transform=target_transform, index=class_index)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True,
                                           num_workers=0, drop_last=True)

# print(len(train_loader))
# print(trainset.labels)
# print(trainset.images.shape)
# print(trainset.labels.shape)
import pylab
for i, data in tqdm(enumerate(train_loader, 0)):
    if i == 0:
        inputs1 , labels1 =data
        # print(len(inputs1))
        # print(len(labels1))
        # print(inputs1.shape)
        # print(inputs1.numpy().shape)

        inputs1 = inputs1.permute(0, 2, 3, 1)
        print(inputs1[0].shape)
        img = inputs1[0].squeeze()
        print(img.shape)
        plt.figure(1)
        plt.imshow(img, cmap="gray")

        # inputs1 = inputs1.reshape(20, 28, 28)
        # print(inputs1[0].shape)
        # plt.imshow(labels1[0].numpy().squeeze())

        plt.figure(2)
        plt.subplot(1, 3, 3)
        plt.subplot(131)
        plt.imshow(inputs1[0].squeeze())
        plt.subplot(132)
        plt.imshow(inputs1[1].squeeze())
        plt.subplot(133)
        plt.imshow(inputs1[2].squeeze())
        plt.show()
        pylab.show()


generator = torch.load('./checkpoints/mnist10_5/task_00_100_model_generator.pkl')
ind = list(range(10))
p_i = list(range(10))
latent_dim = 100
y_onehot = torch.FloatTensor(100, 10)

embed_label_sythesis = []
for _ in range(100):
    np.random.shuffle(ind)
    embed_label_sythesis.append(p_i[ind[0]])
# print(embed_label_sythesis)
embed_label_sythesis = np.asarray(embed_label_sythesis)  # random label
embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
target_lb = embed_label_sythesis.to(torch.long)  #

noise_z = torch.randn(100, latent_dim).cuda()
label = torch.LongTensor(np.array([num for _ in range(10) for num in range(10)])).cuda()
print(label)
# target_lb = torch.full((inputs.size(0), 1), target_label)  # [N,1]
y_onehot.zero_()
y_onehot.scatter_(1, target_lb[:, None], 1)
target_one_hot = y_onehot.cuda()

fake_data = generator(noise_z, target_one_hot)
fake_data = torch.cat((fake_data, fake_data, fake_data),
                          dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
# ## 保存图片
fake_img = fake_data.detach().cpu().permute(0, 2, 3, 1)
fake_img = np.array(fake_img)
current_task = 0
e = 300
# 保存单张图片，将数据还原
fake_img = (fake_img * 0.5 + 0.5)
torchvision.utils.save_image(fake_data * 0.5 + 0.5,
                             './images1/epoch_%d_grid.png' % (e),
                             nrow=10,
                             normalize=True)