import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = 10
nz = 100
nc = 1
noise_z = torch.randn(100, nz, 1, 1, device=device)
datasets = 'mnist'
epoch = 51
log_dir = './checkpoints/mnist10_5_convT'
task = [1,2,3,4,5]
for current_task in task:
    generator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(epoch - 1)))
    # checkpoint = torch.load('./checkpoints/mnist10_5_convT/task_%s_50.pth' % datasets)
    gen_num_class = current_task * 2
    embed_label_sythesis = []
    pre_index = range(gen_num_class)
    ind = list(range(len(pre_index)))
    for _ in range(100):
        np.random.shuffle(ind)
        embed_label_sythesis.append(pre_index[ind[0]])  # 为什么只用旧任务的标签
    print(embed_label_sythesis)
    embed_label_sythesis = np.asarray(embed_label_sythesis)
    embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
    y_onehot = torch.FloatTensor(100, num_class)
    # print(target_lb.shape)
    y_onehot.zero_()
    embed_label_sythesis = embed_label_sythesis.to(torch.long)  #
    y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
    target_one_hot = y_onehot.cuda()
    g_target = target_one_hot.view(100, -1, 1, 1)

    # x = torch.cat((noise_z, g_target), 1)
    fake_data = generator(noise_z, g_target)

    # 如果是单通道图片，那么就转成三通道进行保存
    if nc == 1:
        fake_data = torch.cat((fake_data, fake_data, fake_data), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
    # 保存图片
    data = fake_data.detach().cpu().permute(0, 2, 3, 1)
    data = np.array(data)

    # 保存单张图片，将图片归一化到(0,1)
    data = (data * 0.5 + 0.5)

    plt.imsave('./g/%s/epoch_%d.png' % (datasets, current_task), data[0])
    torchvision.utils.save_image(fake_data, fp='./g/%s/epoch%d_grid.png' % (datasets, current_task),
                                 nrow=10, normalize=True)

