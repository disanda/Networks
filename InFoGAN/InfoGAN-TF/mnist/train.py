import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import Generator, Discriminator, Auxiliary, q_sample
from config import flags
from utils import sample, d_loss, g_loss, info, train_display_img
import tensorlayer as tl
import time
from tqdm import *

(Xtr, ytr), (_, _) = tf.keras.datasets.mnist.load_data()#(60000,28,28)
Xtr = (Xtr/127.5)-1
Xtr = Xtr.reshape([-1, 28, 28, 1]).astype("float32")#(60000,28,28,1)
dataset = tf.data.Dataset.from_tensor_slices(Xtr)
dataset = dataset.shuffle(30000).batch(flags.batch_size)#样本从shuffle池(大小30000)中获取

D = Discriminator([None, 28, 28, 1])#out1:1,out2:7*7*128
G = Generator([None, 74])#out:(-1,28,28,1)
Q = Auxiliary([None, 1024])#out:[cat:10, con1_mu:2, con1_var:2, con2_mu:2, con2_var:2],这里10代表10个变量0-9，2代表两个变量c1,c2

D.train()
G.train()
Q.train()

gen_optimizer = tf.optimizers.Adam(flags.G_learning_rate, 0.5)
dis_optimizer = tf.optimizers.Adam(flags.D_learning_rate, 0.5)


def train_step(imgs):
    noise, z_con1, z_con2, z_cat = sample(flags.batch_size)
    with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        fake_imgs = G(noise)#in:这里是一个拼接过的noise(包含c1,c2,cat)，out:[64,28,28,1]
        real_output, _ = D(imgs)
        fake_output, mid = D(fake_imgs)#[64,1],[64,1024]
        cat, con1_mu, con1_var, con2_mu, con2_var = Q(mid)#Auxiliary:[64,10],[64,2],,,
        fkcat, fkcon1, fkcon2 = cat, q_sample(con1_mu, tf.exp(con1_var)), q_sample(con2_mu, tf.exp(con2_var))
        #q_sample,本代码核心，通过两个c1合成一个c1
        info_loss = info(fkcon1, fkcon2, fkcat, z_con1, z_con2, z_cat)
        gen_loss = g_loss(fake_output)
        dis_loss = d_loss(real_output, fake_output)
        gi = gen_loss+info_loss
        di = dis_loss

    g_grd = gtape.gradient(gi, G.trainable_weights+Q.trainable_weights)#梯度带求梯度，其中gi是y，后者是x(即target和source)
    d_grd = dtape.gradient(di, D.trainable_weights)
    gen_optimizer.apply_gradients(zip(g_grd, G.trainable_weights+Q.trainable_weights))#将计算的梯度应用于变量上
    dis_optimizer.apply_gradients(zip(d_grd, D.trainable_weights))

    return gen_loss, dis_loss, info_loss


def train(dataset, epochs):
    step = 0
    gen_loss = []
    dis_loss = []
    info_loss = []
    for epoch in range(epochs):
        for batch in tqdm(dataset):
            gen, dis, info = train_step(batch)
            gen_loss.append(gen)
            dis_loss.append(dis)
            info_loss.append(info)
            step += 1
            if step % 100 == 0:
                train_display_img(G, step)

        G.save("./models/model{}.h5".format(epoch+1), save_weights=True)
        mg = tf.reduce_mean(gen_loss).numpy()
        md = tf.reduce_mean(dis_loss).numpy()
        mi = tf.reduce_mean(info_loss).numpy()
        print("[{}]\t{:03d}\tGenerator: {:.4f}\tDiscriminator: {:.4f}\tInfo: {:.4f}".format(
            time.strftime('%H:%M:%S', time.localtime(time.time())), epoch+1, mg, md, mi))
    
    plt.figure(figsize=(20, 8))
    plt.plot(gen_loss, label="generator")
    plt.plot(dis_loss, label="discriminator")
    plt.plot(info_loss, label="mutual_info")
    plt.legend()
    plt.suptitle("GAN loss")
    plt.savefig("loss")
    plt.close()


train(dataset, flags.n_epoch)
