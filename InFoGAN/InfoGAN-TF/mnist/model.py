import tensorflow as tf
import tensorlayer as tl
from config import flags


def Generator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    nn = tl.layers.Dense(n_units=1024, b_init=None, W_init=w_init)(ni)#[-1,1024]
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,
                             gamma_init=gamma_init)(nn)
    nn = tl.layers.Dense(n_units=7*7*128, b_init=None, W_init=w_init)(nn)#[-1,6272]
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,gamma_init=gamma_init)(nn)
    nn = tl.layers.Reshape([-1, 7, 7, 128])(nn)#[-1,7,7,128]
    nn = tl.layers.DeConv2d(64, (4, 4), strides=(2, 2), padding="SAME", W_init=w_init)(nn)#[-1,14，14，64]
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,gamma_init=gamma_init)(nn)
    nn = tl.layers.DeConv2d(1, (4, 4), strides=(2, 2), padding="SAME", act=tf.nn.tanh, W_init=w_init)(nn)#[-1,28,28,1]
    return tl.models.Model(inputs=ni, outputs=nn)
#in:[x,74],out:[x,28,28,1]
#74->1024->6272->14,14,64->28,28,1

def Discriminator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)#[-1,28,28,1]
    nn = tl.layers.Conv2d(64, (4, 4), strides=(2, 2), act=lambda x: tl.act.lrelu(x, flags.leaky_rate), padding="SAME", W_init=w_init)(ni)#[-1,14,14,64]
    nn = tl.layers.Conv2d(128, (4, 4), strides=(2, 2), padding="SAME", W_init=w_init)(nn)#[-1,7,7,128]
    nn = tl.layers.BatchNorm2d(decay=0.9, act=lambda x: tl.act.lrelu(x, flags.leaky_rate), gamma_init=gamma_init)(nn)
    nn = tl.layers.Flatten()(nn)
    nn = tl.layers.Dense(n_units=1024, W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=lambda x: tl.act.lrelu(
    x, flags.leaky_rate), gamma_init=gamma_init)(nn)
    mid = nn#[-1,7*7*128]
    nn = tl.layers.Dense(n_units=1, W_init=w_init)(nn)#[-1,1]
    return tl.models.Model(inputs=ni, outputs=[nn, mid])


#con1_mu, tf.exp(con1_var)
def q_sample(mu, var):
    unit = tf.random.normal(shape=mu.shape)
    sigma = tf.sqrt(var)#算var的平方根
    return mu+sigma*unit


def Auxiliary(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    nn = tl.layers.Dense(n_units=128, W_init=w_init)(ni)
    nn = tl.layers.BatchNorm(decay=0.9, act=lambda x: tl.act.lrelu(x, flags.leaky_rate), gamma_init=gamma_init)(nn)

    cat = tl.layers.Dense(n_units=10, W_init=w_init)(nn)
    con1_mu = tl.layers.Dense(n_units=2, W_init=w_init)(nn)
    con1_var = tl.layers.Dense(n_units=2, W_init=w_init)(nn)
    con2_mu = tl.layers.Dense(n_units=2, W_init=w_init)(nn)
    con2_var = tl.layers.Dense(n_units=2, W_init=w_init)(nn)

    return tl.models.Model(inputs=ni, outputs=[cat, con1_mu, con1_var, con2_mu, con2_var])
