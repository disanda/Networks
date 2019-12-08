import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
#优化算法，反向传播更新梯度用的
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

#
mnist = input_data.read_data_sets('../../MNIST', one_hot=True)
mb_size = 64
#就是一轮训练的数量,一个batch
Z_dim = 100
X_dim = mnist.train.images.shape[1]
#mnist.train.images是一个TF的dataset训练集部分，其维度mnist.train.images.shape是(55000，784)，即有55000张图片，每个图片有28*28就是784个维度
# X_dim =784  即mnist.train.images.shape[1]是一张图片的数据量即784

y_dim = mnist.train.labels.shape[1]
#mnist.train.labels.shape是一个(55000,10)的矩阵，代表其标签就是图片具体代表哪个数字，如array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])是代表数字7
# y_dim是10
h_dim = 128
cnt = 0
lr = 1e-3
#这个代表0.001


def xavier_init(size):
    in_dim = size[0]
    #第一个维度:110
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    #开方1/sqrt(55)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
#xavier_init([110,128])就是一个归一化后的随机变量矩阵
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
#(128,784)
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def G(z, c):
    inputs = torch.cat([z, c], 1)
    #cat([a,b],1) a,b按列拼接
    #a:(64,100),b:(64,10),inputs:(64,110)
    h = nn.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1)) #@为装饰器实现,torch里面作用为矩阵相乘
    # (64,110)*(110,128)=(64,128)
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    # (64,128)*(128,784)=(64,784)
    return X
#换句话说生成器受c控制

""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
#(784+10,128)
bxh = Variable(torch.zeros(h_dim), requires_grad=True)
#128
Why = xavier_init(size=[h_dim, 1])
#(128,1)
bhy = Variable(torch.zeros(1), requires_grad=True)


def D(X, c):
    inputs = torch.cat([X, c], 1)
    #X:(64,784),c:(64,10),input:(64,794)
    h = nn.relu(inputs @ Wxh + bxh.repeat(inputs.size(0), 1))
    #h:(64，794)*(794,128)=(64,128)
    y = nn.sigmoid(h @ Why + bhy.repeat(h.size(0), 1))
    #y=(64,128)*(128,1)=(64,1)
    return y
#看图片X是否和标签C对应


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params


""" ===================== TRAINING ======================== """


def reset_grad():
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())


G_solver = optim.Adam(G_params, lr=1e-3)
#BP跟新梯度用的是Adam,更新的参数是G_params
D_solver = optim.Adam(D_params, lr=1e-3)

ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))


for it in range(100000):
    # Sample data
    z = Variable(torch.randn(mb_size, Z_dim))
    #(64，110)
    X, c = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    #[64,784]
    c = Variable(torch.from_numpy(c.astype('float32')))
    #[64,10]

    # Dicriminator forward-loss-backward-update
    G_sample = G(z, c)
    #z:(64,100),c:(64,10),G_sample:(64,784)
    D_real = D(X, c)
    #X:(64,784),c:(64,10),D_real:(64,1)
    D_fake = D(G_sample, c)


    D_loss_real = nn.binary_cross_entropy(D_real, ones_label)
    #交叉熵,其值越小说明两参数(分布)越相似
    print("D_loss_real："+D_loss_real)
    D_loss_fake = nn.binary_cross_entropy(D_fake, zeros_label)
    print("D_loss_fake："+D_loss_fake)
    D_loss = D_loss_real + D_loss_fake

    D_loss.backward()
    D_solver.step()

    # Housekeeping - reset gradient
    reset_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(mb_size, Z_dim))
    #z:(64,100) 有64个z，每一个z有100个值，最后可以生成一个784即28*28的图片
    G_sample = G(z, c)
    #G_sample:(64,784)
    D_fake = D(G_sample, c)
    # c是图片的真实标签

    G_loss = nn.binary_cross_entropy(D_fake, ones_label)
    #判别器结果分布和全对结果分布

    G_loss.backward()
    #代表BP
    G_solver.step()
    #代表优化器BP后更新一次参数

    # Housekeeping - reset gradient
    reset_grad()

    c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
    #c:(64,10),设置生成器要生成的样本标签
    c[:, 7] = 1.
   #c[:, 8] = 1
    c = Variable(torch.from_numpy(c))

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.data.numpy(), G_loss.data.numpy()))
        samples = G(z, c).data.numpy()[:16]
        #通过切割第一维，取前16张图片
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)