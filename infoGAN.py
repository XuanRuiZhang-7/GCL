import torch, time, os, pickle, itertools
import torchvision

import myutils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import mydataloader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

import pdb
from spectral import SpectralNorm


class info_generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(info_generator, self).__init__()
        self.input_dim = input_dim      # 随机噪声
        self.output_dim = output_dim    # channel
        self.input_size = input_size    # 图像尺寸
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label) 类标签
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)  固定部分

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        # ngf = 64
        # nz = self.input_dim + self.len_discrete_code + self.len_continuous_code
        # self.deconv = nn.Sequential(
        #     nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True), nn.BatchNorm2d(ngf * 8),  # 2
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True), nn.BatchNorm2d(ngf * 4),  # 4
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True), nn.BatchNorm2d(ngf * 2),  # 8
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(ngf * 2, self.output_dim, 4, 2, 1, bias=True),
        #     nn.Tanh()  # (N,nc, 128,128)
        # )
        myutils.initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)  # bs*[128*7*7]
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        # x = x.view(input.shape[0], -1, 1, 1)

        x = self.deconv(x)

        return x


class info_discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(info_discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            # nn.Sigmoid(),
        )
        myutils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)    # bs*128*7*7
        # pdb.set_trace()
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))   # bs*[128*7*7]
        x = self.fc(x)  # bs*13
        a = F.sigmoid(x[:, self.output_dim])  # bs*1
        b = x[:, self.output_dim:self.output_dim + self.len_continuous_code]    # bs*2
        c = x[:, self.output_dim + self.len_continuous_code:]   # bs*10

        return a, b, c


class infoGAN(object):
    def __init__(self, args, SUPERVISED=True):
        # parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size   # fanshionmnist-28
        self.z_dim = args.z_dim
        self.SUPERVISED = SUPERVISED        # if it is true, label info is directly used for code
        self.len_discrete_code = 10         # categorical distribution (i.e. label)
        # self.len_discrete_code = 100         # categorical distribution (i.e. label)
        self.len_continuous_code = 2        # gaussian distribution (e.g. rotation, thickness)
        self.sample_num = self.len_discrete_code ** 2
        print(self.input_size)

        # load dataset
        self.data_loader = mydataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]
        print(data.shape)    # torch.Size([64, 3, 32, 32])


        # networks init
        self.G = info_generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size, len_discrete_code=self.len_discrete_code, len_continuous_code=self.len_continuous_code)
        # pdb.set_trace()
        self.D = info_discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size, len_discrete_code=self.len_discrete_code, len_continuous_code=self.len_continuous_code)
        opt_RMSprop = torch.optim.RMSprop(self.G.parameters(), lr=0.0008, alpha=0.9)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        # self.G_optimizer = optim.Adam(self.G.parameters(), clipvalue=1.0,   # 在优化器中使用梯度裁剪（限制梯度值的范围）
        #                               lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.scheduler_G = StepLR(self.G_optimizer, step_size=args.gan_lr_decay_step, gamma=0.3)
        self.scheduler_D = StepLR(self.D_optimizer, step_size=args.gan_lr_decay_step, gamma=0.3)
        self.scheduler_info = StepLR(self.info_optimizer, step_size=args.gan_lr_decay_step, gamma=0.3)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        myutils.print_network(self.G)
        myutils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.len_discrete_code):
            self.sample_z_[i * self.len_discrete_code] = torch.rand(1, self.z_dim)  # 均匀分布
            # self.sample_z_[i * self.len_discrete_code] = torch.randn(1, self.z_dim)#  正态分布
            for j in range(1, self.len_discrete_code):
                self.sample_z_[i * self.len_discrete_code + j] = self.sample_z_[i * self.len_discrete_code]

        temp = torch.zeros((self.len_discrete_code, 1))
        for i in range(self.len_discrete_code):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.len_discrete_code):
            temp_y[i * self.len_discrete_code: (i + 1) * self.len_discrete_code] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        # np.random.shuffle(self.sample_y_)
        self.sample_c_ = torch.zeros((self.sample_num, self.len_continuous_code))

        # manipulating two continuous code 相同噪声、相同标签、连续的c
        self.sample_z2_ = torch.rand((1, self.z_dim)).expand(self.sample_num, self.z_dim)
        # self.sample_z2_ = torch.randn((1, self.z_dim)).expand(self.sample_num, self.z_dim)
        self.sample_y2_ = torch.zeros(self.sample_num, self.len_discrete_code)
        self.sample_y2_[:, 0] = 1

        temp_c = torch.linspace(-1, 1, 10)
        # temp_c = torch.linspace(-1, 1, 100)
        self.sample_c2_ = torch.zeros((self.sample_num, 2))
        for i in range(self.len_discrete_code):
            for j in range(self.len_discrete_code):
                self.sample_c2_[i*self.len_discrete_code+j, 0] = temp_c[i]
                self.sample_c2_[i*self.len_discrete_code+j, 1] = temp_c[j]

        if self.gpu_mode:
            self.sample_z_, self.sample_y_, self.sample_c_, self.sample_z2_, self.sample_y2_, self.sample_c2_ = \
                self.sample_z_.cuda(), self.sample_y_.cuda(), self.sample_c_.cuda(), self.sample_z2_.cuda(), \
                self.sample_y2_.cuda(), self.sample_c2_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        # resume = True
        # if resume:
        #     print('==> Resuming from checkpoint..')
        #     D = torch.load('D:\Project\PycharmProject\GImgR\models\/fashion-mnist\infoGAN\infoGAN_D.pkl' )
        #     G = torch.load('D:\Project\PycharmProject\GImgR\models\/fashion-mnist\infoGAN\infoGAN_G.pkl' )
        #     # t = torch.load('D:\Project\PycharmProject\GImgR\models\/fashion-mnist\infoGAN\infoGAN_history.pkl' )
        #     self.G = G
        #     self.D = D
        #     start_epoch = 50

        for epoch in range(self.epoch):
            self.G.train()
            # self.scheduler_info.step()
            # self.scheduler_D.step()
            # self.scheduler_G.step()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                # z_ = torch.randn((self.batch_size, self.z_dim))
                if self.SUPERVISED == True:
                    y_disc_ = torch.zeros((self.batch_size, self.len_discrete_code)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                else:
                    y_disc_ = torch.from_numpy(
                        np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                              size=[self.batch_size])).type(torch.FloatTensor)
                y_cont_ = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, 2))).type(torch.FloatTensor)

                if self.gpu_mode:
                    x_, z_, y_disc_, y_cont_ = x_.cuda(), z_.cuda(), y_disc_.cuda(), y_cont_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real, _, _ = self.D(x_)   # torch.Size([64])
                D_real = D_real.reshape(-1,1)
                # print(D_real.shape)
                # print(self.y_real_.shape)   # torch.Size([64, 1])
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_, y_cont_, y_disc_)
                # print(G_.shape)
                D_fake, _, _ = self.D(G_)
                # D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                D_fake_loss = self.BCE_loss(D_fake.reshape(-1,1), self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                # update G network
                # if iter % 5 == 0:
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_cont_, y_disc_)
                D_fake, D_cont, D_disc = self.D(G_)

                G_loss = self.BCE_loss(D_fake.reshape(-1, 1), self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # information loss
                disc_loss = self.CE_loss(D_disc.detach(), torch.max(y_disc_, 1)[1])
                # disc_loss = self.CE_loss(D_disc, y_disc_.to(torch.long))
                cont_loss = self.MSE_loss(D_cont.detach(), y_cont_)
                info_loss = disc_loss + cont_loss
                info_loss = 0.5 * info_loss
                self.train_hist['info_loss'].append(info_loss.item())
                # torch.autograd.set_detect_anomaly(True)   # 异常检测
                # with torch.autograd.detect_anomaly():
                #     info_loss.backward(retain_graph=True)
                info_loss.requires_grad_(True)
                info_loss.backward()

                self.info_optimizer.step()


                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item(), info_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))   # 保存生成图像

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()     # 保存模型
        myutils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                   self.epoch)
        myutils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont',
                                   self.epoch)
        self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        self.loss_plot_G(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ style by class """
        samples = self.G(self.sample_z_, self.sample_c_, self.sample_y_)    # torch.Size([100, 3, 28, 28])
        print(samples.shape)
        # if self.gpu_mode:
        #     samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        # else:
        #     samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        # myutils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #                   self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')
        fake_data = samples[:image_frame_dim * image_frame_dim, :, :, :]
        # fake_data = torch.from_numpy(fake_data)
        # fake_data = fake_data.permute(0, 3, 1, 2)
        # print(fake_data.shape)  # (100, 28, 28, 1)
        if fake_data.shape[1] == 1:
            fake_data = torch.cat((fake_data, fake_data, fake_data),dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
        torch_resize = transforms.Resize([64, 64])  # 定义Resize类对象
        fake_data = torch_resize(fake_data)
        torchvision.utils.save_image(fake_data,
                                     '%s/%s/%s/%s_epoch%03d_grid.png' % (self.result_dir, self.dataset, self.model_name, self.model_name, epoch),
                                     nrow=10,
                                     normalize=True)

        """ manipulating two continous codes """
        samples = self.G(self.sample_z2_, self.sample_c2_, self.sample_y2_)
        # print(samples.shape)
        # if self.gpu_mode:
        #     samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        # else:
        #     samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        # myutils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #                   self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_cont_epoch%03d' % epoch + '.png')
        fake_data = samples[:image_frame_dim * image_frame_dim, :, :, :]
        # fake_data = torch.from_numpy(fake_data)
        # fake_data = fake_data.permute(0, 3, 1, 2)
        print(fake_data.shape)  # (100, 28, 28, 1)
        if fake_data.shape[1] ==1:
            fake_data = torch.cat((fake_data, fake_data, fake_data),dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)        print(fake_data.shape)  # (100, 28, 28, 1)
        torchvision.utils.save_image(fake_data,
                                     # self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png',
                                     '%s/%s/%s/%s_cont_epoch%03d_grid.png' % (self.result_dir, self.dataset,self.model_name, self.model_name, epoch),
                                     nrow=10,
                                     normalize=True)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        # y2 = hist['G_loss']
        # y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        # plt.plot(x, y2, label='G_loss')
        # plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)

    def loss_plot_G(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['G_loss']))

        y1 = hist['G_loss']
        y2 = hist['info_loss']

        plt.plot(x, y1, label='G_loss')
        plt.plot(x, y2, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_G_loss.png')
        plt.savefig(path)