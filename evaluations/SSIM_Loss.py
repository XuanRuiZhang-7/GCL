import os
import sys

import numpy as np
import torchvision

from PIL import Image
import torch
# from paddle.optimizer import Adam
import cv2
# import paddle
# from paddle_msssim import ssim, ms_ssim

import torch
import torch.nn.functional as F
from math import exp
import matplotlib.pyplot as plt

from torch.autograd import Variable

import pytorch_ssim
import numpy as np
# 创建一维高斯分布向量



from tqdm import tqdm

with tqdm(range(10)) as t:
    ...
    t.update()
    print(t.format_interval(t.format_dict['elapsed']))
def imread(img_path):
    img = cv2.imread(img_path)
    # return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)
    return torch.LongTensor(img.transpose(2, 0, 1))


img1 = imread('D:\Project\PycharmProject\GImgR\generated_fake\mnist\epoch_8.png')   # 真实图像
img2 = imread('D:\Project\PycharmProject\GImgR\generated_fake\mnist\epoch_10.png')  # 生成图像
img1 = img1.view(1,3,128,128)
img2 = img2.view(1,3,128,128)
ssim_0 = pytorch_ssim.ssim(img1, img2)

print('[SSIM] simga_0: %f  ' % (ssim_0))
#
# ms_ssim_0 = ms_ssim(simga_0, simga_n)
# print('[MS-SSIM] simga_0: %f simga_50: %f simga_100: %f' % (ms_ssim_0))

generator = torch.load('D:\Project\PycharmProject\GImgR\checkpoints\mnist_linear_ssim\/task_02_100_model_generator.pkl')
sample_num = 100
noise_z = torch.randn(sample_num, 100).cuda()
onehot = torch.LongTensor(sample_num, 10)
len_discrete_code = 10
temp = torch.zeros((len_discrete_code, 1))
for i in range(len_discrete_code):
    temp[i, 0] = i
temp_y = torch.zeros((sample_num, 1))
for i in range(len_discrete_code):
    temp_y[i * len_discrete_code: (i + 1) * len_discrete_code] = temp
sample_y_ = torch.zeros((sample_num, 10)).scatter_(1, temp_y.type(torch.LongTensor), 1)
noise_z,sample_y_ = noise_z.cuda(),sample_y_.cuda()
fake_data = generator(noise_z, sample_y_)
fake_data = torch.cat((fake_data, fake_data, fake_data), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)


sample_num = 100
noise_z = torch.randn(sample_num, 100).cuda()
onehot = torch.LongTensor(sample_num, 10)
ind = list(range(2))
p_i = list(range(2))
sample_label = []
for _ in range(sample_num):
    np.random.shuffle(ind)
    sample_label.append(p_i[ind[0]])
sample_label = np.asarray(sample_label)  # random label
sample_label = torch.from_numpy(sample_label).to(torch.long)
# # sample_label = torch.LongTensor(np.array([num for _ in range(10) for num in range(10)]))
# sample_label = torch.LongTensor(np.array([num for _ in range(10) for num in range(len(pre_index) + num_class_per_task)]))
onehot.zero_().scatter_(1, sample_label.reshape(-1, 1), 1)
target_one_hot = onehot.cuda()

fake_data = generator(noise_z, target_one_hot)
fake_data = torch.cat((fake_data, fake_data, fake_data), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
# ## 保存图片
# fake_img = fake_data.detach().cpu().permute(0, 2, 3, 1)
# fake_img = np.array(fake_img)
# # 保存单张图片，将数据还原

epoch = 10000
torchvision.utils.save_image(fake_data * 0.5 + 0.5,
                             'D:\Project\PycharmProject\GImgR\/results\mnist\linear\/task_10_epoch_%d_grid.png' % (epoch),
                             # nrow=len(pre_index) + num_class_per_task,   # 每行图片个数
                             nrow=10,  # 每行图片个数
                             normalize=True)

loss_type = 'ssim'
assert loss_type in ['ssim', 'msssim']

# if loss_type == 'ssim':
#     loss_obj = pytorch_ssim.ssim(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
# else:
#     loss_obj = ms_ssim(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

ssim_value = pytorch_ssim.ssim(img1, img2)
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()
print("Initial ssim:", ssim_loss(fake_data, fake_data))

optimizer = torch.optim.Adam([img2], lr=0.01)
img1 = Variable(img1, requires_grad=False)
img2 = Variable(img2, requires_grad=False)
step = 0
while ssim_value < 0.95:
    step += 1
    # optimizer.clear_grad()  # 清空梯度
    optimizer.zero_grad()  # 置为0
    loss = ssim_loss(img1, img2)
    (1 - loss).backward()
    optimizer.step()
    ssim_value = loss.item()
    if step % 10 == 0:
        print('step: %d %s: %f' % (step, loss_type, ssim_value))
#

