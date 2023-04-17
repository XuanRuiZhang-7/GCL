import os

"""
    在需要批量执行py文件的目录下，输入执行文件的名称，运行此文件即可
"""

file_ = ["train1", "train2", "train3", "train4", "train5"]
# file_ = ["0331","0332","0333","0334"]
os.system("python train_mnist.py -data mnist -G_type convT -num_class 10 -nb_cl_fg 2 -num_task 4 "
              "-BatchSize 100  -tradeoff 1 -epochs 101 -lr 1e-3  -epochs_gan 301  -lr_decay_step 20 -gan_lr 0.0002 "
              "-gan_lr_decay_step 150 -log_dir mnist10_5_linear -dir D:/Dataset -gpu 0")
