
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
from tqdm import tqdm
from copy import deepcopy
import torchvision
cudnn.benchmark = True

G_loss = 100 \
          + 1e-3 * 10 * 10
print(G_loss)

loss_cls = 100
nb_cl_fg = 50
num_class_per_task = 10
current_task = 5
loss_cls /= nb_cl_fg // num_class_per_task + current_task
print(loss_cls)
a=[0,1,2,3,4,5,6,7,8,9]
print(a[:5])
print(a[5:])

num_class=100
latent_dim = 200
BatchSize = 128
epochs_gan = 501
embed_sythesis = []
embed_label_sythesis = []
# pre_index = range(50)  # 生成全部类别的图像*****
pre_index = [1]  # 生成指定类别的图像***效果不佳**
print(pre_index)
ind = list(range(len(pre_index)))
print(ind)
log_dir = 'cifar100_10_tradeoff2'
generator = torch.load('./checkpoints/cifar100_10_tradeoff2/task_00_500_model_generator.pkl')
model = torch.load('./checkpoints/cifar100_10_tradeoff2/task_00_200_model.pkl')
for _ in range(BatchSize):
    np.random.shuffle(ind)
    embed_label_sythesis.append(pre_index[ind[0]])
embed_label_sythesis = np.asarray(embed_label_sythesis)  # random label
embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
y_onehot = torch.FloatTensor(BatchSize, num_class)
print(embed_label_sythesis)

y_onehot.zero_()
# embed_label_sythesis = torch.tensor(embed_label_sythesis, dtype=torch.long)  #
embed_label_sythesis = embed_label_sythesis.to(torch.long)  #
y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
syn_label_pre = y_onehot.cuda()
print(syn_label_pre)

z = torch.Tensor(np.random.normal(0, 1, (BatchSize, latent_dim))).cuda()
img_sythesis = generator(z, syn_label_pre)  # embed is img here

# ## 保存图片
fake_img = img_sythesis.detach().cpu().permute(0, 2, 3, 1)
fake_img = np.array(fake_img)
# 保存单张图片，将数据还原
fake_img = (fake_img * 0.5 + 0.5)
print(fake_img.shape)
print(fake_img[0].shape)
import cv2
plt.figure(2)
plt.subplot(1, 3, 3)
plt.subplot(131)
plt.imshow(fake_img[0])
plt.subplot(132)
plt.imshow(fake_img[1])
plt.subplot(133)
plt.imshow(fake_img[2])
plt.show()

# plt.imsave('./t_generated_fake/GFR_%d/epoch_%d.png' % (current_task, epoch), fake_img[0])

img_feature = model(img_sythesis)
soft_feat_syt = model.embed(img_feature)

soft_feat_syt = soft_feat_syt.cpu().detach().numpy()
img_estimate = np.argmax(soft_feat_syt, axis=1)  # 预测的标签
print(img_estimate)

num=0
for i in range(len(embed_label_sythesis)):
    if img_estimate[i] == embed_label_sythesis[i]:
        num+=1

print(num)
acc = num/ float(len(embed_label_sythesis))
print(acc)
