# 导入相关模块
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import struct
import gzip
from PIL import Image
import pdb


class MNIST(Dataset):  # 继承Dataset

    # def __init__(self, , transform=None):  # __init__是初始化该类的一些基础参数
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    img_size = 28
    # The images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size
    filename_x_train = "train-images-idx3-ubyte"
    filename_y_train = "train-labels-idx1-ubyte"
    filename_x_test = "t10k-images-idx3-ubyte"
    filename_y_test = "t10k-labels-idx1-ubyte"


#     filename_x_train = "train-images-idx3-ubyte.gz"
#     filename_y_train = "train-labels-idx1-ubyte.gz"
#     filename_x_test = "t10k-images-idx3-ubyte.gz"
#     filename_y_test = "t10k-labels-idx1-ubyte.gz"
     

    def __init__(self, root, train=True,
                     transform=None, target_transform=None,
                     download=False, index=None):
        self.root_dir = os.path.expanduser(root)  # 文件目录
        # print(root)
        # print(self.root_dir)
        self.transform = transform  # 变换
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.images = os.listdir(self.root_dir)  # 目录里的所有文件

        self.data = []
        self.targets = []

        if self.train:
            data_file = self.filename_x_train
            target_file = self.filename_y_train
        else:
            data_file = self.filename_x_test
            target_file = self.filename_y_test

        self.images, self.targets = self.load_data(data_file, target_file, index)
        # self.img = self._load_images(filename=data_file, index=index) / 255.0
        # self.targets = self._load_cls(filename=target_file, index=index)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        img, targets = self.images[index], self.targets[index]

        # image_index = self.images[index]  # 根据索引index获取该图片
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        # img = io.imread(img_path)  # 读取该图片
        # label = img_path.split('\\')[-1].split('.')[0]
        # # 根据该图片的路径名获取该图片的label，具体根据路径名进行分割。我这里是"E:\\Python Project\\Pytorch\\dogs-vs-cats\\train\\cat.0.jpg"，所以先用"\\"分割，选取最后一个为['cat.0.jpg']，然后使用"."分割，选取[cat]作为该图片的标签
        # sample = {'image': img, 'label': label}  # 根据图片和标签创建字典
        #
        # if self.transform:
        #     sample = self.transform(sample)  # 对样本进行变换
        # return sample  # 返回该样本

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, targets

    def load_data(self, data_file, target_file, index):
        labels_path = os.path.join(self.root_dir, target_file)
        img_path = os.path.join(self.root_dir, data_file)

        with gzip.open(labels_path, 'rb') as f:  # rb表示的是读取二进制数据
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        with gzip.open(img_path, 'rb') as imgpath:
            img = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(labels), 28, 28)
            # print(img.shape)

        # images_flat = img.reshape(-1, self.img_size_flat)
        # print(index)
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == labels)[0]
            if data_tmp==[]:
                data_tmp = img[ind_cl]
                targets_tmp = labels[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp,img[ind_cl]))
                targets_tmp = np.hstack((targets_tmp,labels[ind_cl]))
        # print(data_tmp.shape)
        data, targets = data_tmp, targets_tmp

        return data, targets

    def load_mnist(path, index, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % (kind))
        images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % (kind))
        # labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        # images_path=os.path.join(path,'%s-images.idx3-ubyte'%kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
            # print(len(labels))
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            # labels=np.array(labels)
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels


from torch import nn


# G_type = linear
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
        # out = torch.cat((noise, self.label_emb(label)), -1)
        out = torch.cat((noise, label), -1)
        img = self.model(out)     # torch.Size([64, 784])
        img = img.view(img.size(0), 1, 28, 28)     # torch.Size([64, 1, 32, 32])
        return img


# batch_size = 100
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
        img = img.view(img.size(0), -1)     # [bs,28,28]->torch.Size([bs, 784])
        # y_onehot = torch.FloatTensor(img.shape[0],10)
        # y_onehot.zero_()
        labels = torch.tensor(label, dtype=torch.long)  #
        # y_onehot.scatter_(1, labels[:, None], 1)
        # syn_label = y_onehot
        # print(syn_label.shape)
        x = torch.cat((img, labels), -1)     # torch.Size([bs, 794])
        x = self.model(x)   # torch.Size([bs, 1])
        return x


# G_type = convt
class Mnist_Generator_ConvT(nn.Module):
        def __init__(self, channel, n_classes=10, nz=100, ngf=64):
            super(Mnist_Generator_ConvT, self).__init__()
            nc = channel
            # 生成器                             #(N,nz, 1,1)
            self.model = nn.Sequential(nn.ConvTranspose2d(nz+n_classes, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                                 nn.Tanh()  # (N,nz, 128,128)
                                 )

        def forward(self, noise, label):
            # out = torch.cat((noise, self.label_emb(label)), -1)
            noise = noise.view(noise.size(0), -1, 1, 1)
            label = label.view(noise.size(0), -1, 1, 1)
            out = torch.cat((noise, label), 1)  # torch.Size([bs, 110, 1, 1])
            img = self.model(out)  # torch.Size([bs, 1, 128, 128])
            return img


class Mnist_Discriminator_ConvT(nn.Module):
        def __init__(self, channel, n_classes=200, nz=100, ndf=64):
            super(Mnist_Discriminator_ConvT, self).__init__()
            nc = channel
            # 判别器             #(N,nc, 128,128)
            self.model = nn.Sequential(nn.Conv2d(nc+n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                                 nn.Flatten(),  # (N,1)
                                 nn.Sigmoid()
                                 )

        def forward(self, img, label):
            # print('discriminator')
            # print(img.size(0))
            # print(label.shape)
            labels = torch.tensor(label, dtype=torch.long)  #
            labels = labels.view(img.size(0), -1, 1, 1)
            labels = labels.repeat(1, 1, img.size(2), img.size(3))

            x = torch.cat((img, labels), 1)  # torch.Size([bs, 11, 128, 128])
            score = self.model(x)  # torch.Size([bs, 1])

            return score


# G_type = info
class Mnist_Generator_info(nn.Module):
        def __init__(self, channel, n_classes=10, nz=100, ngf=64):
            super(Mnist_Generator_info, self).__init__()
            nc = channel
            # 生成器                             #(N,nz, 1,1)
            self.model = nn.Sequential(nn.ConvTranspose2d(nz+n_classes, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                                 nn.Tanh()  # (N,nz, 128,128)
                                 )

        def forward(self, noise, label):
            # out = torch.cat((noise, self.label_emb(label)), -1)
            noise = noise.view(noise.size(0), -1, 1, 1)
            label = label.view(noise.size(0), -1, 1, 1)
            out = torch.cat((noise, label), 1)  # torch.Size([bs, 110, 1, 1])
            img = self.model(out)  # torch.Size([bs, 1, 128, 128])
            return img


class Mnist_Discriminator_info(nn.Module):
        def __init__(self, channel, n_classes=200, nz=100, ndf=64):
            super(Mnist_Discriminator_info, self).__init__()
            nc = channel
            # 判别器             #(N,nc, 128,128)
            self.model = nn.Sequential(nn.Conv2d(nc+n_classes, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # (N,1,1,1)
                                 nn.Flatten(),  # (N,1)
                                 nn.Sigmoid()
                                 )

        def forward(self, img, label):
            # print('discriminator')
            # print(img.size(0))
            # print(label.shape)
            labels = torch.tensor(label, dtype=torch.long)  #
            labels = labels.view(img.size(0), -1, 1, 1)
            labels = labels.repeat(1, 1, img.size(2), img.size(3))

            x = torch.cat((img, labels), 1)  # torch.Size([bs, 11, 128, 128])
            score = self.model(x)  # torch.Size([bs, 1])

            return score
