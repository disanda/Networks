from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

import util
from variable import LatentVariable, build_latent_variables

class Noise(nn.Module):
    def __init__(self, use_noise: float, sigma: float = 0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma
        self.device = util.current_device()
    def forward(self, x):
        if self.use_noise:
            return (x + self.sigma * torch.empty(x.size(), device=self.device, requires_grad=False).normal_())
        return x

class Generator(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()
        self.latent_vars = latent_vars
        self.dim_input = sum(map(lambda x: x.cdim, latent_vars.values()))
        #aaa=map(lambda x: x.cdim, latent_vars.values())，映射函数，得到一个map对象，list(aaa)后是一个列表[64,10,1,1],即map映射出cdim这个属性的数据
        #dim_input=64+10+1+1=76
        ngf = 64
        self.device = util.current_device()
        # main layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dim_input, ngf * 8, 4, 1, 0, bias=False),#输入77，输出512
            nn.BatchNorm2d(ngf * 8),#输入512,输出512
            nn.ReLU(True),#输入512，输出512
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),#输入512，输出256
            nn.BatchNorm2d(ngf * 4),#输入256，输出256
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, x):
        return self.main(x)
    def forward_dummy(self) -> torch.Tensor:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = torch.empty(shape, device=self.device).normal_()
        return self.forward(dummy_tensor)
    def sample_latent_vars(self, batchsize: int) -> Dict[str, torch.Tensor]:
    #核心函数，用于产生输入变量zs,其是一个字典，键为z,cat,c1,c2,值为对应随机数
        zs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, var in self.latent_vars.items():
            #name是键(包括:z,c1,c1,c2)，
            zs[name] = var.prob.sample([batchsize, var.dim])
            #var.prob是分布对象，其方法sample([batchsize, var.dim])是采样出的随机值
            zs[name] = zs[name].to(self.device)
            print(name)
            print(zs[name].size())
        return zs
    def infer(self, zs: List[torch.Tensor]) -> torch.Tensor:
    #将输入变量z,cat,c1,c2拼接起来
        z = torch.cat(zs, dim=1)#(100,76)
        z = z.unsqueeze(-1).unsqueeze(-1)#(100,76,1,1)
        x = self.forward(z)
        return x
    @property
    def module(self) -> nn.Module:
        return self

class Discriminator(nn.Module):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__()
        self.configs = configs
        ndf = 64
        self.device = util.current_device()
        use_noise: bool = configs["use_noise"]
        noise_sigma: float = configs["noise_sigma"]
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),#channel=1,pix:64*64
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4,channel:64*8,pix:4*4
        )
    def forward(self, x):
        return self.main(x)
    def forward_dummy(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    #给一个shape，输出(-1,512,4,4)
        shape = (2, 1, 64, 64)
        dummy_tensor = torch.empty(shape, device=self.device).normal_()
        return self.forward(dummy_tensor)
    @property
    def module(self) -> nn.Module:
        return self

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            # state size. 1 x 4 x 4
        )
    def forward(self, x):
        return self.main(x)
    @property
    def module(self) -> nn.Module:
        return self

class QHead(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. 128 x 2 x 2
        )
        # generate each head module from latent variable
        self.convs = nn.ModuleDict()
        for name, var in latent_vars.items():
            if var.kind == "z":
                continue
            if var.prob_name == "normal":
                self.convs[name] = nn.Conv2d(ndf * 2, var.cdim * 2, 1)#2维
            else:
                self.convs[name] = nn.Conv2d(ndf * 2, var.cdim, 1)#10维，数字分类
    def forward(self, x):
        mid = self.main(x)
        ys: Dict[str, torch.Tensor] = {}
        for name, conv in self.convs.items():
            ys[name] = conv(mid).squeeze()
        return ys
    @property
    def module(self) -> nn.Module:
        return self

if __name__ == "__main__":
    configs = util.load_yaml("../configs/debug.yaml")
    latent_vars = build_latent_variables(configs["latent_variables"])#得到一个输入变量，即z,cat,c1,c2组成的字典
    g = Generator(latent_vars)#g是一个网络结构
    zs = g.sample_latent_vars(2)#意味着有100个变量,结构是字典，字典的值是z,cat,c1,c2的拼接
    for k, v in zs.items():
        print(k, v.shape)
    x = g.infer(list(zs.values()))#拼接变量做网络的输入[100,76,1,1]，输出为[100,1,64,64]
    print("x:", x.shape)
    d = Discriminator(configs["models"]["dis"])#判别器模型
    d_head, q_head = DHead(), QHead(latent_vars)
    mid = d(x)
    y, c = d_head(mid), q_head(mid)
    print("mid:", mid.shape)
    print("y:", y.shape)
    print("c:", {k: v.size() for k, v in c.items()})
    print(c['c2'])
    print(c['c2'].max(dim=0))#tensor的max()函数按dim=0输出该行中最大值即其下标(两个列表)
    print('``````````````````')
    print(c['c2'].max(dim=1))#按列分别输出该行最大元素值，及最大行的索引
    print('``````````````````')
    print(c['c3'])
    print('``````````````````')
    f=torch.tensor(([1.,0.2],[15.,250.]))
    layer = Noise(float)
    print(f)
    print(layer(f))
    print('``````````````````')
    dis = Discriminator(configs["models"]["dis"])
    aaa=dis.forward_dummy()
    print(aaa.shape)
    #sprint(aaa)


# z torch.Size([100, 64])
# c1 torch.Size([100, 10])
# c2 torch.Size([100, 1])
# c3 torch.Size([100, 1])
#------------------------------
# x: torch.Size([100, 1, 64, 64])
# mid: torch.Size([100, 512, 4, 4])
# y: torch.Size([100, 1, 4, 4])
# c: {'c1': torch.Size([100, 10]), 'c2': torch.Size([100, 2]), 'c3': torch.Size([100])} ,在debug.yaml中c3是uniform
#c1中10个元素最大的值为真实标签one-hot