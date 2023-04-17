# coding=utf-8
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
from models.resnet2 import Generator, Discriminator, ClassifierMLP, ModelCNN, Generator_IMG, Discriminator_IMG
import torch.autograd as autograd
import scipy.io as sio
from CIFAR100 import CIFAR100,CIFAR10
from MNIST import MNIST
from tqdm import tqdm
from MNIST import Mnist_Generator,Mnist_Discriminator


cudnn.benchmark = True
from copy import deepcopy
import torchvision


def to_binary(labels, args):
    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(len(labels), args.num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.cpu()[:, None], 1)
    code_binary = y_onehot.cuda()
    return code_binary


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return model


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def compute_gradient_penalty(D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    org_shape = real_samples.shape

    interpolates = (alpha * real_samples.view(org_shape[0], -1)
                    + ((1 - alpha) * fake_samples.view(org_shape[0], -1))).requires_grad_(True)
    # d_interpolates, _ = D(interpolates.view(org_shape), syn_label)
    d_interpolates = D(interpolates.view(org_shape), syn_label)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = \
        autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True,
                      retain_graph=True,
                      only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


def compute_prototype(model, data_loader, number_samples=200):
    model.eval()
    count = 0
    embeddings = []
    embeddings_labels = []
    terminate_flag = min(len(data_loader), number_samples)
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if i > terminate_flag:
                break
            count += 1
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            embed_feat = model(inputs)
            embeddings_labels.append(labels.numpy())
            embeddings.append(embed_feat.cpu().numpy())

    embeddings = np.asarray(embeddings)
    embeddings = np.reshape(embeddings, (embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]))
    embeddings_labels = np.asarray(embeddings_labels)
    embeddings_labels = np.reshape(embeddings_labels, embeddings_labels.shape[0] * embeddings_labels.shape[1])
    labels_set = np.unique(embeddings_labels)
    class_mean = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        embeddings_tmp = embeddings[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_std.append(np.std(embeddings_tmp, axis=0))
    prototype = {'class_mean': class_mean, 'class_std': class_std, 'class_label': class_label}

    return prototype


def get_cycle_sizes(start_percent=0.25, end_percent=1.0, multiplier=0.5, epochs=200):
    S = []
    n = start_percent
    S.append(n)
    t = 0
    while sum(S) < epochs:
        if (n == start_percent) or ((S[t - 1] < S[t]) and (n != end_percent)):
            n = min((n * (1 / multiplier)), end_percent)
        else:
            n = max((n * (multiplier)), start_percent)
        S.append(n)
        t += 1
    return S


def get_train_list(model, trainset):
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=False,
                                                    num_workers=args.nThreads, drop_last=True)
        loss_cls = []
        for i, data in enumerate(train_loader, 0):
                inputs1, labels1 = data
                inputs1, labels1 = inputs1.cuda(), labels1.cuda()
                embed_feat = model(inputs1)
                soft_feat = model.embed(embed_feat)
                labels = torch.tensor(labels1, dtype=torch.long)  # 2022/5/27
                loss = torch.nn.CrossEntropyLoss(reduce=False)(soft_feat, labels)
                # loss_cls.extend(loss.tolist())
                loss_cls.extend((1.0 / loss).tolist())
        loss_cls[:] = loss_cls[:]/np.sum(loss_cls)
        # range_size = _get_range_size(train_loader)
        # pdb.set_trace()
        return loss_cls


def train_task(args, trainset, testloader, current_task, prototype={}, pre_index=0):
    num_class_per_task = (args.num_class - args.nb_cl_fg) // args.num_task
    task_range = list(range(args.nb_cl_fg + (current_task - 1) * num_class_per_task,
                            args.nb_cl_fg + current_task * num_class_per_task))
    if num_class_per_task == 0:
        pass  # JT
    else:
        old_task_factor = args.nb_cl_fg // num_class_per_task + current_task - 1    # 10+current-1
    log_dir = os.path.join(args.ckpt_dir, args.log_dir)
    mkdir_if_missing(log_dir)

    sys.stdout = logging.Logger(os.path.join(log_dir, 'log_task{}.txt'.format(current_task)))
    tb_writer = SummaryWriter(log_dir)
    display(args)
    # One-hot encoding or attribute encoding
    if 'imagenet' in args.data:
        model = models.create('resnet18_imagenet', pretrained=False, feat_dim=args.feat_dim, embed_dim=args.num_class)
        channel = 3

    elif 'cifar' in args.data:
        model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim, embed_dim=args.num_class)
        channel = 3

    elif args.data == 'cifar' :
        model = models.create('resnet18_cifar', pretrained=False, feat_dim=args.feat_dim, embed_dim=args.num_class)
        channel = 3

    elif 'mnist' in args.data:
        model = models.create('resnet18_mnist', pretrained=False, feat_dim=args.feat_dim, embed_dim=args.num_class)
        channel = 1

    if current_task > 0:
        model = torch.load(
            os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model.pkl' % int(args.epochs - 1)))
        model_old = deepcopy(model)
        model_old.eval()
        model_old = freeze_model(model_old)


    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

    loss_mse = torch.nn.MSELoss(reduction='sum')

    # Loss weight for gradient penalty used in W-GAN
    lambda_gp = args.lambda_gp
    lambda_lwf = args.gan_tradeoff
    # Initialize generator and discriminator
    if current_task == 0:
        # generator = Generator(feat_dim=args.feat_dim,latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, class_dim=args.num_class)
        # discriminator = Discriminator(feat_dim=args.feat_dim,hidden_dim=args.hidden_dim, class_dim=args.num_class)
        if 'mnist' in args.data:
            generator = Mnist_Generator()
            discriminator = Mnist_Discriminator()
        else:
            generator = Generator_IMG(channel=channel, n_classes=args.num_class, nz=args.latent_dim, ngf=64)
            discriminator = Discriminator_IMG(channel=channel, n_classes=args.num_class, ndf=64)
    else:
        generator = torch.load(os.path.join(log_dir,
                                            'task_' + str(current_task - 1).zfill(2) + '_%d_model_generator.pkl' % int(
                                                args.epochs_gan - 1)))
        discriminator = torch.load(os.path.join(log_dir, 'task_' + str(current_task - 1).zfill(2) + '_%d_model_discriminator.pkl' % int(args.epochs_gan - 1)))
        generator_old = deepcopy(generator)
        generator_old.eval()
        generator_old = freeze_model(generator_old)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    scheduler_G = StepLR(optimizer_G, step_size=args.gan_lr_decay_step, gamma=0.3)
    scheduler_D = StepLR(optimizer_D, step_size=args.gan_lr_decay_step, gamma=0.3)

    # Y_onehot is used to generate one-hot encoding
    y_onehot = torch.FloatTensor(args.BatchSize, args.num_class)

    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = False

    ###############################################################Feature extractor training####################################################
    if current_task > 0:
        model = model.eval()
    # Curriculum Learning
    data_sizes = get_cycle_sizes(start_percent=0.256, end_percent=1.0, multiplier=0.5, epochs=args.epochs_gan + 1)
    print(data_sizes)
    range_size = len(trainset.targets)
    if current_task > 0:
        scores = get_train_list(model, trainset)
    else:
        rand = np.random.dirichlet(np.ones(range_size), size=1)
        scores = rand[0]

    for epoch in range(args.epochs):
        loss_log = {'C/loss': 0.0,
                    'C/loss_aug': 0.0,
                    'C/loss_cls': 0.0}
        # scheduler.step()
        from torch.utils.data import Sampler
        # index = np.random.choice(range(0, range_size), int(data_sizes[epoch]*range_size), p=scores, replace=False)
        sampler = torch.utils.data.WeightedRandomSampler(scores, int(data_sizes[epoch + 1] * range_size),
                                                         replacement=False)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=False,
                                                   # sampler=sampler,
                                                   num_workers=args.nThreads, drop_last=True)  #
        for i, data in tqdm(enumerate(train_loader, 0)):
            inputs1, labels1 = data
            inputs1, labels1 = inputs1.cuda(), labels1.cuda()

            loss = torch.zeros(1).cuda()
            loss_cls = torch.zeros(1).cuda()
            loss_aug = torch.zeros(1).cuda()
            optimizer.zero_grad()

            inputs, labels = inputs1, labels1  # !  inputs=[bs,3,32] label=[bs]

            # ## Classification loss
            embed_feat = model(inputs)
            if current_task == 0:
                soft_feat = model.embed(embed_feat)
                # labels = torch.tensor(labels, dtype=torch.long)  #
                labels = labels.to(torch.long)  #

                loss_cls = torch.nn.CrossEntropyLoss()(soft_feat, labels)
                loss += loss_cls
            else:
                embed_feat_old = model_old(inputs)

            ### Feature Extractor Loss
            if current_task > 0:
                loss_aug = torch.dist(embed_feat, embed_feat_old, 2)    # 当前任务数据的蒸馏（L2范数）
                loss += args.tradeoff * loss_aug * old_task_factor      # old_task_factor = 10+current-1

            ### Replay and Classification Loss
            if current_task > 0:
                embed_sythesis = []
                embed_label_sythesis = []
                ind = list(range(len(pre_index)))  # ind [0-> pre_index)

                if args.mean_replay:
                    for _ in range(args.BatchSize):
                        np.random.shuffle(ind)
                        tmp = prototype['class_mean'][ind[0]] + np.random.normal() * prototype['class_std'][ind[0]]
                        embed_sythesis.append(tmp)
                        embed_label_sythesis.append(prototype['class_label'][ind[0]])
                    embed_sythesis = np.asarray(embed_sythesis)
                    embed_label_sythesis = np.asarray(embed_label_sythesis)
                    embed_sythesis = torch.from_numpy(embed_sythesis).cuda()
                    embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                else:
                    for _ in range(args.BatchSize):
                        np.random.shuffle(ind)
                        embed_label_sythesis.append(pre_index[ind[0]])
                    embed_label_sythesis = np.asarray(embed_label_sythesis)  # random label
                    embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                    y_onehot.zero_()
                    # embed_label_sythesis = torch.tensor(embed_label_sythesis, dtype=torch.long)  #
                    embed_label_sythesis = embed_label_sythesis.to(torch.long)  #

                    y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
                    syn_label_pre = y_onehot.cuda()

                    z = torch.Tensor(np.random.normal(0, 1, (args.BatchSize, args.latent_dim))).cuda()
                    embed_sythesis = generator(z, syn_label_pre)  # embed is img here

                # ======================== org code ================================#
                # embed_sythesis = torch.cat((embed_feat,embed_sythesis))
                # embed_label_sythesis = torch.cat((labels,embed_label_sythesis.cuda()))
                # soft_feat_syt = model.embed(embed_sythesis)
                # # real samples,   exemplars,      synthetic samples
                # #           batch_size1       batch_size2

                # ========================== new code for img gen ==================#
                img_sythesis = torch.cat((inputs, embed_sythesis), 0)
                embed_label_sythesis = torch.cat((labels, embed_label_sythesis.cuda()))
                embed_sythesis = model(img_sythesis)
                soft_feat_syt = model.embed(embed_sythesis)

                # ==================================================================#

                batch_size1 = inputs1.shape[0]
                batch_size2 = embed_feat.shape[0]

                loss_cls = torch.nn.CrossEntropyLoss()(soft_feat_syt[:batch_size1],
                                                       embed_label_sythesis[:batch_size1])  # 新任务数据

                loss_cls_old = torch.nn.CrossEntropyLoss()(soft_feat_syt[batch_size2:],
                                                           embed_label_sythesis[batch_size2:])  # 生成数据

                loss_cls += loss_cls_old * old_task_factor
                loss_cls /= args.nb_cl_fg // num_class_per_task + current_task      # loss_cls/(5+current_task)
                loss += loss_cls

            loss.backward()
            optimizer.step()

            loss_log['C/loss'] += loss.item()
            loss_log['C/loss_cls'] += loss_cls.item()
            loss_log['C/loss_aug'] += args.tradeoff * loss_aug.item() if args.tradeoff != 0 else 0
            del loss_cls
            if epoch == 0 and i == 0:
                print(50 * '#')
        scheduler.step()
        # 验证
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                model.eval()
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                soft_outputs = model.embed(outputs)

                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(soft_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        print('[Metric Epoch %05d]\t Total Loss: %.3f \t LwF Loss: %.3f \tLwF Loss*old_task_factor: %.3f \t'
              % (epoch + 1, loss_log['C/loss'], loss_log['C/loss_aug'], loss_log['C/loss_aug']*old_task_factor))
        for k, v in loss_log.items():
            if v != 0:
                tb_writer.add_scalar('Task {} - Classifier/{}'.format(current_task, k), v, epoch + 1)

        if epoch == args.epochs - 1:
            torch.save(model, os.path.join(log_dir, 'task_' + str(
                current_task).zfill(2) + '_%d_model.pkl' % epoch))
    ################################################################## W-GAN Training stage####################################################
    model = model.eval()
    for p in model.parameters():  # set requires_grad to False
        p.requires_grad = False
    for p in generator.parameters():  # set requires_grad to False
        p.requires_grad = True
    criterion_softmax = torch.nn.CrossEntropyLoss().cuda()

    if current_task != args.num_task:
        for epoch in range(args.epochs_gan):
            loss_log = {'D/loss': 0.0,
                        'D/new_rf': 0.0,
                        'D/new_lbls': 0.0,
                        'D/new_gp': 0.0,
                        'D/prev_rf': 0.0,
                        'D/prev_lbls': 0.0,
                        'D/prev_gp': 0.0,
                        'G/loss': 0.0,
                        'G/new_rf': 0.0,
                        'G/new_lbls': 0.0,
                        'G/prev_rf': 0.0,
                        'G/prev_mse': 0.0,
                        'G/new_classifier': 0.0,
                        'E/kld': 0.0,
                        'E/mse': 0.0,
                        'E/loss': 0.0}
            scheduler_D.step()
            scheduler_G.step()
            sampler = torch.utils.data.WeightedRandomSampler(scores, int(data_sizes[epoch + 1] * range_size),
                                                             replacement=False)

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=False,
                                                       # sampler=sampler,
                                                       num_workers=args.nThreads, drop_last=True)  # 2022/11/18
            for i, data in tqdm(enumerate(train_loader, 0)):    # tqdm 进度条工具
                for p in discriminator.parameters():
                    p.requires_grad = True

                inputs, labels = data

                inputs = Variable(inputs.cuda())

                ############################# Train Disciminator###########################
                optimizer_D.zero_grad()

                # org
                # real_feat = model(inputs)
                # z = torch.Tensor(np.random.normal(0, 1, (args.BatchSize, 3, 32, 32))).cuda()  # z for img

                y_onehot.zero_()
                labels = torch.tensor(labels, dtype=torch.long)  #
                y_onehot.scatter_(1, labels[:, None], 1)
                syn_label = y_onehot.cuda()

                # org
                # fake_feat = generator(z, syn_label)
                # fake_validity, _ = discriminator(fake_feat, syn_label)
                # real_validity, disc_real_acgan = discriminator(real_feat, syn_label)
                # ================================ orginal code ====================================================#
                # fake_validity, _               = discriminator(fake_feat, syn_label)
                # real_validity, disc_real_acgan = discriminator(real_feat, syn_label)
                #
                # # Adversarial loss
                # d_loss_rf = -torch.mean(real_validity) + torch.mean(fake_validity)
                # gradient_penalty = compute_gradient_penalty(discriminator, real_feat, fake_feat, syn_label).mean()
                # d_loss_lbls = criterion_softmax(disc_real_acgan, labels.cuda())
                # d_loss = d_loss_rf + lambda_gp * gradient_penalty
                # =================================   CGAN Training for D   ========================================#

                # change for img generator
                real_feat = inputs  # here real_feat means orginal img inputs

                real_validity = discriminator(real_feat, syn_label)
                # cgan loss
                positive_label = torch.full((real_feat.size(0), 1), 1.0).cuda()
                # print(real_validity.shape)
                # print(positive_label.shape)
                d_loss_cgan_p = torch.nn.BCELoss()(real_validity, positive_label)
                #d_loss_cgan_p.backward()
                z = torch.Tensor(np.random.normal(0, 1, (args.BatchSize, args.latent_dim))).cuda()
                fake_feat = generator(z, syn_label)  # fake_feat = fake_img

                fake_validity = discriminator(fake_feat.detach(), syn_label)
                negitave_label = torch.full((real_feat.size(0), 1), 0.0).cuda()
                d_loss_cgan_n = torch.nn.BCELoss()(fake_validity, negitave_label)
                #d_loss_cgan_n.backward()

                d_loss_cgan = d_loss_cgan_p + d_loss_cgan_n

                # Adversarial loss
                d_loss_rf = -torch.mean(real_validity) + torch.mean(fake_validity)  # validity 0 假 1 真
                # gradient_penalty = compute_gradient_penalty(discriminator, real_feat, fake_feat, syn_label).mean()
                # d_loss = 0.3 * (d_loss_rf + lambda_gp * gradient_penalty) + d_loss_cgan * 0.7
                # d_loss = d_loss_rf + lambda_gp * gradient_penalty + d_loss_cgan
                d_loss = d_loss_cgan
                # ====================================================================================================#

                d_loss.backward()
                optimizer_D.step()      # 更新鉴别器
                loss_log['D/loss'] += d_loss.item()
                loss_log['D/new_rf'] += d_loss_rf.item()
                loss_log['D/new_lbls'] += 0  # !!!
                # loss_log['D/new_gp'] += gradient_penalty.item() if lambda_gp != 0 else 0
                # del d_loss_rf, d_loss_lbls
                del d_loss_rf
                ############################# Train Generaator###########################
                # Train the generator every n_critic steps
                if i % args.n_critic == 0:
                    for p in discriminator.parameters():
                        p.requires_grad = False
                        ############################# Train GAN###########################
                    optimizer_G.zero_grad()
                    # Generate a batch of images
                    fake_feat = generator(z, syn_label)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    # fake_validity, disc_fake_acgan = discriminator(fake_feat, syn_label)
                    fake_validity = discriminator(fake_feat.cuda(), syn_label)

                    if current_task == 0:
                        loss_aug = 0 * torch.sum(fake_validity)
                    else:
                        ind = list(range(len(pre_index)))
                        embed_label_sythesis = []
                        for _ in range(args.BatchSize):
                            np.random.shuffle(ind)
                            embed_label_sythesis.append(pre_index[ind[0]])  # 为什么只用旧任务的标签

                        embed_label_sythesis = np.asarray(embed_label_sythesis)
                        embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                        y_onehot.zero_()
                        # embed_label_sythesis = torch.tensor(embed_label_sythesis, dtype=torch.long)  #
                        embed_label_sythesis = embed_label_sythesis.to(torch.long)  #

                        y_onehot.scatter_(1, embed_label_sythesis[:, None], 1)
                        syn_label_pre = y_onehot.cuda()

                        pre_feat = generator(z, syn_label_pre)
                        pre_feat_old = generator_old(z, syn_label_pre)
                        loss_aug = loss_mse(pre_feat, pre_feat_old)     # 回放对齐（均方误差损失）


                    # g_loss_rf = -torch.mean(fake_validity)            # orginal g_loss_rf is -torch.mean(fake_validity)

                    positive_label = torch.full((real_feat.size(0), 1), 1.0).cuda()
                    g_loss_rf = torch.nn.BCELoss()(fake_validity, positive_label)


                    # g_loss_lbls = criterion_softmax(disc_fake_acgan, labels.cuda())
                    g_loss = g_loss_rf \
                             + lambda_lwf * old_task_factor * loss_aug
                    # g_loss = g_loss_rf
                    loss_log['G/loss'] += g_loss.item()
                    loss_log['G/new_rf'] += g_loss_rf.item()
                    loss_log['G/new_lbls'] += 0  # !
                    loss_log['G/new_classifier'] += 0  # !
                    loss_log['G/prev_mse'] += loss_aug.item() if lambda_lwf != 0 else 0
                    # del g_loss_rf, g_loss_lbls
                    del g_loss_rf
                    g_loss.backward()
                    optimizer_G.step()

                    if i % 10 == 0:
                        print('[GAN Epoch %05d]\t Dp Loss: %.5f Dn Loss: %.5f \t G Loss: %.5f \t LwF Loss: %.5f' % (
                            epoch + 1, d_loss_cgan_p.item(), d_loss_cgan_n.item(), g_loss.item(),
                            loss_log['G/prev_rf']))
                    # 每50个epoch保存图片
                    if epoch % 2 == 0 and i == 0:
                        # target_labe=l = 1
                        # 生成指定target_label的图片
                        ind = list(range(len(pre_index)+num_class_per_task))
                        p_i = list(range(len(pre_index)+num_class_per_task))
                        embed_label_sythesis = []
                        for _ in range(100):
                            np.random.shuffle(ind)
                            embed_label_sythesis.append(p_i[ind[0]])
                        # print(embed_label_sythesis)
                        embed_label_sythesis = np.asarray(embed_label_sythesis)  # random label
                        embed_label_sythesis = torch.from_numpy(embed_label_sythesis)
                        y_onehot.zero_()
                        target_lb = embed_label_sythesis.to(torch.long)  #

                        noise_z = torch.randn(100, args.latent_dim, 1, 1).cuda()
                        if 'mnist' in args.data:
                            noise_z = torch.randn(100, args.latent_dim).cuda()


                        # target_lb = torch.full((inputs.size(0), 1), target_label)  # [N,1]
                        y_onehot.zero_()
                        y_onehot.scatter_(1, target_lb[:, None], 1)
                        target_one_hot = y_onehot.cuda()

                        fake_data = generator(noise_z, target_one_hot)
                        if channel == 1:
                            fake_data = torch.cat((fake_data, fake_data, fake_data),
                                                  dim=1)  # fake_data（N,1,H,W）->(N,3,H,W)
                        # ## 保存图片
                        fake_img = fake_data.detach().cpu().permute(0, 2, 3, 1)
                        fake_img = np.array(fake_img)

                        # 保存单张图片，将数据还原
                        fake_img = (fake_img * 0.5 + 0.5)
                        plt.imsave('./generated_fake/GFR_%d/epoch_%d.png' % (current_task, epoch), fake_img[0])
                        # plt.imsave('./generated_fake/GFR_%d/epoch_%d.png' % (current_task, epoch), fake_img[0])

                        torchvision.utils.save_image(fake_data[:] * 0.5 + 0.5,
                                                     './generated_fake/GFR_%d/epoch_%d_grid.png' % (
                                                     current_task, epoch),
                                                     nrow=10,
                                                     normalize=True)

            print('[GAN Epoch %05d]\t D Loss: %.3f \t G Loss: %.3f \t LwF Loss: %.3f' % (
                epoch + 1, loss_log['D/loss'], loss_log['G/loss'], loss_log['G/prev_rf']))
            for k, v in loss_log.items():
                if v != 0:
                    tb_writer.add_scalar('Task {} - GAN/{}'.format(current_task, k), v, epoch + 1)

            if epoch == args.epochs_gan - 1:
                torch.save(generator, os.path.join(log_dir, 'task_' + str(
                    current_task).zfill(2) + '_%d_model_generator.pkl' % epoch))
                torch.save(discriminator, os.path.join(log_dir, 'task_' + str(
                    current_task).zfill(2) + '_%d_model_discriminator.pkl' % epoch))

    tb_writer.close()

    prototype = compute_prototype(model, train_loader)  # !
    return prototype


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generative Feature Replay Training')

    # task setting
    parser.add_argument('-data', default='cifar100', required=True, help='path to Data Set')
    parser.add_argument('-num_class', default=100, type=int, metavar='n', help='dimension of embedding space')
    parser.add_argument('-nb_cl_fg', type=int, default=50, help="Number of class, first group")
    parser.add_argument('-num_task', type=int, default=2, help="Number of Task after initial Task")

    # method parameters
    parser.add_argument('-mean_replay', action='store_true', help='Mean Replay')
    parser.add_argument('-tradeoff', type=float, default=1.0, help="Feature Distillation Loss")

    # basic parameters
    parser.add_argument('-load_dir_aug', default='', help='Load first task')
    parser.add_argument('-ckpt_dir', default='checkpoints', help='checkpoints dir')
    parser.add_argument('-dir', default='/data/datasets/featureGeneration/', help='data dir')
    parser.add_argument('-log_dir', default=None, help='where the trained models save')
    parser.add_argument('-name', type=str, default='tmp', metavar='PATH')

    parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
    parser.add_argument('-nThreads', '-j', default=4, type=int, metavar='N', help='number of data loading threads')

    # hyper-parameters
    parser.add_argument('-BatchSize', '-b', default=128, type=int, metavar='N', help='mini-batch size')  # 128
    parser.add_argument('-lr', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-lr_decay', type=float, default=0.1, help='Decay learning rate')
    parser.add_argument('-lr_decay_step', type=float, default=100, help='Decay learning rate every x steps')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-weight-decay', type=float, default=2e-4)

    # hype-parameters for W-GAN
    parser.add_argument('-gan_tradeoff', type=float, default=1e-3, help="learning rate of new parameters")
    parser.add_argument('-gan_lr', type=float, default=1e-4, help="learning rate of new parameters")
    parser.add_argument('-lambda_gp', type=float, default=10.0, help="learning rate of new parameters")
    parser.add_argument('-n_critic', type=int, default=5, help="learning rate of new parameters")
    parser.add_argument('-gan_lr_decay_step', type=float, default=200, help='Decay learning rate every x steps')

    parser.add_argument('-latent_dim', type=int, default=200, help="learning rate of new parameters")
    parser.add_argument('-feat_dim', type=int, default=512, help="learning rate of new parameters")
    parser.add_argument('-hidden_dim', type=int, default=512, help="learning rate of new parameters")

    # training parameters
    parser.add_argument('-epochs', default=201, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-epochs_gan', default=1001, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-seed', default=1993, type=int, metavar='N', help='seeds for training process')
    parser.add_argument('-start', default=0, type=int, help='resume epoch')

    args = parser.parse_args()

    # Data
    print('==> Preparing data..')

    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        traindir = os.path.join(args.dir, 'ILSVRC12_256', 'train')

    if args.data == 'cifar100':
        transform_train = transforms.Compose([
            # transforms.Resize((32, 32)),
########
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     # GFR orginal normalize param
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
######
            transforms.Resize(128),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        traindir = args.dir + '/cifar'
    if args.data == 'cifar10':
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                ########
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),     # GFR orginal normalize param
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ######
                # transforms.Resize(128),
                # transforms.RandomCrop(128),
                # transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            traindir = args.dir + '/cifar'
            classes_lable = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.data == 'mnist':    # 28*28
        transform_train = transforms.Compose([
            # transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        traindir = args.dir + '/mnist'

    num_classes = args.num_class
    num_task = args.num_task
    num_class_per_task = (num_classes - args.nb_cl_fg) // num_task

    random_perm = list(range(num_classes))  # multihead fails if random permutaion here
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prototype = {}

    if args.mean_replay:
        args.epochs_gan = 2

    for i in range(args.start, num_task + 1):
        print("-------------------Get started--------------- ")
        print("Training on Task " + str(i))
        if i == 0:
            # pre_index = 0
            pre_index = []
            class_index = random_perm[:args.nb_cl_fg]
        else:
            pre_index = random_perm[:args.nb_cl_fg + (i - 1) * num_class_per_task]
            class_index = random_perm[
                          args.nb_cl_fg + (i - 1) * num_class_per_task:args.nb_cl_fg + (i) * num_class_per_task]

        if args.data == 'cifar100':
            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR100(root=traindir, train=True, download=True, transform=transform_train,
                                target_transform=None, index=class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True,
                                                       num_workers=args.nThreads, drop_last=True)
        elif args.data == 'cifar10':
            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = CIFAR10(root=traindir, train=True, download=True, transform=transform_train,
                               target_transform=None, index=class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True,
                                                       num_workers=args.nThreads, drop_last=True)
            testset = CIFAR10(root='D:\Dataset\cifar', train=False, download=True, transform=transform_train,
                              target_transform=None, index=random_perm)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0,
                                                      drop_last=False)
        elif args.data == 'mnist':
            # train_dataset = torchvision.datasets.MNIST(root=traindir, train=True, transform=transform_train, download=True)

            np.random.seed(args.seed)
            target_transform = np.random.permutation(num_classes)
            trainset = MNIST(root=traindir, train=True, download=False, transform=transform_train,
                             target_transform=target_transform, index=class_index)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize, shuffle=True,
                                                       num_workers=args.nThreads, drop_last=True)
            print(class_index)
            print(len(train_loader))

            # (x_train, t_train), (x_test, t_test) = MNIST.load_mnist(path=traindir, kind='train', index=class_index)
            # x_train, t_train, x_test, t_test = map(torch.tensor, (x_train, t_train, x_test, t_test))
            # # 数据包装
            # trainset = torch.utils.data.TensorDataset(x_train, t_train)
            # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.BatchSize)

        else:
            trainfolder = ImageFolder(traindir, transform_train, index=class_index)
            train_loader = torch.utils.data.DataLoader(
                trainfolder, batch_size=args.BatchSize,
                shuffle=True,
                drop_last=True, num_workers=args.nThreads)

        prototype_old = prototype
        prototype = train_task(args, trainset, test_loader, i, prototype=prototype, pre_index=pre_index)

        if args.start > 0:
            pass
            # Currently it only works for our PrototypeLwF method
        else:
            if i >= 1:
                # Increase the prototype as increasing number of task
                for k in prototype.keys():
                    prototype[k] = np.concatenate((prototype[k], prototype_old[k]), axis=0)
