import argparse
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import loss
import util
from dataset import new_fashion_mnist_dataset, new_mnist_dataset
from logger import Logger
from model import DHead, Discriminator, Generator, QHead
from trainer import Trainer
from variable import build_latent_variables

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def worker_init_fn(worker_id: int):
    random.seed(worker_id)

def create_optimizer(models: List[nn.Module], lr: float, decay: float):
#优化器，输入是模型、学习率和延迟率,输出是一个
    params: List[torch.Tensor] = []
    for m in models:
        params += list(m.parameters())#m.parameters迭代器,保存着当前网络结构(module)的各层参数，需要通过梯度下降去调整
    return optim.Adam(params, lr=lr, betas=(0.5, 0.999), weight_decay=decay)#优化器，输入是要梯度更新的参数，学习率及其他配置betas默认值是[0.9,0.999],weight_decay默认值是0

def main():
    parser = argparse.ArgumentParser()#参数对象
    parser.add_argument(
        "--config",
        "-c",
        default="../configs/mnist.yaml",
        help="training configuration file",
    )
    args = parser.parse_args()#解析参数对象,通过属性名去索引参数
    configs = util.load_yaml(args.config)#args.config实际就是个路径名
    dataset_name = configs["dataset"]["name"]
    dataset_path = Path(configs["dataset"]["path"]) / dataset_name
    # prepare dataset
    if dataset_name == "mnist":
        dataset = new_mnist_dataset(dataset_path)
    elif dataset_name == "fashion-mnist":
        dataset = new_fashion_mnist_dataset(dataset_path)
    else:
        raise NotImplementedError
    dataloader = DataLoader(
        dataset,
        batch_size=configs["dataset"]["batchsize"],
        num_workers=configs["dataset"]["n_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    # prepare models
    latent_vars = build_latent_variables(configs["latent_variables"])#输入是一个字典，[z,c1:,c2,c3,c4],值是自定义的latantvariable类
    gen, dis = Generator(latent_vars), Discriminator(configs["models"]["dis"])#
    dhead, qhead = DHead(), QHead(latent_vars)#普通的D,和详细的D
    models = {"gen": gen, "dis": dis, "dhead": dhead, "qhead": qhead}#各个模型对象并入一个字典
    # prepare optimizers
    opt_gen = create_optimizer([gen, qhead], **configs["optimizer"]["gen"])
    opt_dis = create_optimizer([dis, dhead], **configs["optimizer"]["dis"])
    opts = {"gen": opt_gen, "dis": opt_dis}
    # prepare directories
    log_path = Path(configs["log_path"])
    log_path.mkdir(parents=True, exist_ok=True)
    tb_path = Path(configs["tensorboard_path"])
    tb_path.mkdir(parents=True, exist_ok=True)
    # initialize logger
    logger = Logger(log_path, tb_path)#参数为两个路径
    # initialize losses
    losses = {"adv": loss.AdversarialLoss(), "info": loss.InfoGANLoss(latent_vars)}
    # start training
    trainer = Trainer(dataloader, latent_vars, models, opts, losses, configs["training"], logger)
    #第二个参数latent_vars:输入数据对象latentvarible组成的字典
    #第三个参数models:模型对象组成的字典
    #第四个参数opts:优化器对象组成的字典
    #第五个参数losses:损失函数对象组成的字典
    #第六个参数configs["traning"]:关于训练设置参数的字典
    #第七个参数logger:自定义的Logger类
    trainer.train()

if __name__ == "__main__":
    main()
