import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from MNIST import MNIST
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_class = 10
nz = 100
nc = 1
noise_z = torch.randn(100, nz, 1, 1, device=device)
datasets = 'mnist'
epoch = 51
log_dir = './checkpoints/mnist10_5_convT'
dir = 'D:/Dataset'
transform_train = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
traindir = dir + '/mnist'
random_perm = list(range(num_class ))
nb_cl_fg = 2
num_class_per_task = 2
task = range(5)

for current_task in task:
    generator = torch.load(os.path.join(log_dir, 'task_' + str(current_task).zfill(2) + '_%d_model_generator.pkl' % int(epoch - 1)))
    # checkpoint = torch.load('./checkpoints/mnist10_5_convT/task_%s_50.pth' % datasets)

    if current_task == 0:
        # pre_index = 0
        pre_index = []
        class_index = random_perm[:nb_cl_fg]
    else:
        pre_index = random_perm[:nb_cl_fg + (current_task - 1) * num_class_per_task]
        class_index = random_perm[
                      nb_cl_fg + (current_task - 1) * num_class_per_task:nb_cl_fg + (current_task) * num_class_per_task]

    gen_num_class = current_task * 2
    embed_label_sythesis = []
    np.random.seed(143)
    target_transform = np.random.permutation(10)
    trainset = MNIST(root=traindir, train=True, download=False, transform=transform_train,
                     target_transform=target_transform, index=class_index)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,num_workers=0, drop_last=True)
    for i, data in enumerate(train_loader, 0):
        if i == 0:
            inputs1, labels1 = data
            inputs1, labels1 = inputs1.cuda(), labels1.cuda()
            print(inputs1.shape)    # torch.Size([100, 1, 128, 128])

            # 如果是单通道图片，那么就转成三通道进行保存
            if nc == 1:
                fake_data = torch.cat((inputs1, inputs1, inputs1), dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
            # 保存图片
            data = fake_data.detach().cpu().permute(0, 2, 3, 1)
            data = np.array(data)

            # 保存单张图片，将图片归一化到(0,1)
            data = (data * 0.5 + 0.5)
            print(data.shape)

            plt.imsave('./g/%s/epoch_%d.png' % (datasets, current_task * 10), data[0])
            torchvision.utils.save_image(fake_data,
                                         fp='./g/%s/epoch%d_grid.png' % (datasets, current_task * 10),
                                         nrow=10,
                                         normalize=True)


