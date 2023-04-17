import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim
import os
import numpy as np
import torchvision.transforms as transforms
from MNIST import MNIST


# 设置超参数
from MNIST import MNIST

batch_size = 100
learning_rate = 0.0002
# epochsize = 90
epochsize = 180
sample_dir = "images3"

# 创建生成图像的目录
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# 生成器结构
class Mnist_Generator(nn.Module):
    def __init__(self):
        super(Mnist_Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(

            nn.Linear(110, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, label):
        out = torch.cat((noise, self.label_emb(label)), -1)
        img = self.model(out)     # torch.Size([64, 784])
        img = img.view(img.size(0), 1, 28, 28)     # torch.Size([64, 1, 32, 32])
        return img


# 鉴别器结构
class Mnist_Discriminator(nn.Module):
    def __init__(self):
        super(Mnist_Discriminator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(

            nn.Linear(794, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        img = img.view(img.size(0), -1)     # torch.Size([100, 784])
        y_onehot = torch.FloatTensor(batch_size,10)

        y_onehot.zero_()
        labels = torch.tensor(label, dtype=torch.long)  #
        y_onehot.scatter_(1, labels[:, None], 1)
        syn_label = y_onehot
        # print(label)
        # print(img.shape)
        # print(syn_label.shape)
        x = torch.cat((img, syn_label), -1)     # torch.Size([100, 794])

        x = self.model(x)   # torch.Size([100, 1])
        return x

import torchvision

# 训练集下载
class_index = list(range(10))
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
target_transform = np.random.permutation(10)

# mnist_traindata = datasets.MNIST('D:/Dataset/mnist', train=True, transform=transform_train, download=False)

mnist_traindata = MNIST(root='D:/Dataset/mnist', train=True, download=False, transform=transform_train,
                             target_transform=target_transform, index=class_index)

mnist_train = torch.utils.data.DataLoader(mnist_traindata, batch_size=batch_size, shuffle=True, pin_memory=True)

# GPU加速
# device = torch.device('cuda')
# torch.cuda.set_device(0)

G = Mnist_Generator()
D = Mnist_Discriminator()

# 导入之前的训练模型
G.load_state_dict(torch.load('G_plus.ckpt'))
D.load_state_dict(torch.load('D_plus.ckpt'))

# 设置优化器与损失函数,二分类的时候使用BCELoss较好,BCEWithLogitsLoss是自带一层Sigmoid
# criteon = nn.BCEWithLogitsLoss()
criteon = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

# 开始训练
print("start training")
for epoch in range(epochsize):

    D_loss_total = 0
    G_loss_total = 0
    total_num = 0

    # 这里的RealImageLabel是没有用上的
    for batchidx, (realimage, realimage_label) in enumerate(mnist_train):

        # realimage = realimage.to(device)
        realscore = torch.ones(realimage.size(0), 1)   # value：1 torch.Size([128, 1])
        fakescore = torch.zeros(realimage.size(0), 1)   # value：0 torch.Size([128, 1])

        # 随机sample出噪声与标签，生成假图像
        z = torch.randn(realimage.size(0), 100)
        fakeimage_label = torch.LongTensor(np.random.randint(0, 10, realimage.size(0)))
        fakeimage = G(z, fakeimage_label)

        # 训练鉴别器————总的损失为两者相加
        d_realimage_loss = criteon(D(realimage, realimage_label), realscore)
        d_fakeimage_loss = criteon(D(fakeimage, fakeimage_label), fakescore)
        D_loss = d_realimage_loss + d_fakeimage_loss

        # 参数训练三个步骤
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # 计算一次epoch的总损失
        D_loss_total += D_loss

        # 训练生成器————损失只有一个
        # 上一次的梯度信息以消除，重新生成假图像
        fakeimage = G(z, fakeimage_label)
        G_loss = criteon(D(fakeimage, fakeimage_label), realscore)

        # 参数训练三个步骤
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # 计算一次epoch的总损失
        G_loss_total += G_loss

        # 打印相关的loss值
        if batchidx % 200 == 0:
            print("batchidx:{}/{}, D_loss:{}, G_loss:{}".format(batchidx, len(mnist_train), D_loss, G_loss))

    # 打印一次训练的loss值
    print('Epoch:{}/{}, D_loss:{}, G_loss:{}'.format(epoch, epochsize, D_loss_total / len(mnist_train),
                                                                   G_loss_total / len(mnist_train)))

    # 保存生成图像
    z = torch.randn(batch_size, 100)
    label = torch.LongTensor(np.array([num for _ in range(10) for num in range(10)]))
    save_image(G(z, label).data, os.path.join(sample_dir, 'images-{}.png'.format(epoch + 61)), nrow=10, normalize=True)

    # 保存网络结构
    torch.save(G.state_dict(), 'G_plus.ckpt')
    torch.save(D.state_dict(), 'D_plus.ckpt')
