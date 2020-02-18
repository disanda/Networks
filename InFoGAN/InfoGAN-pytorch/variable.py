from collections import OrderedDict
from typing import Any, Dict, Sequence,List

import torch
import torch.distributions as dist

class LatentVariable(object):
#该对象的通过prob参数生产一个随机数赋值给self.prob
    def __init__(self, name: str, kind: str, prob: str, dim: int, **kwargs: Any):
    #多参数函数，可以通过**kwargs参数传入一个字典
        self.name: str = name
        self.kind: str = kind  # "z", "c"
        self.dim: int = dim
        self.cdim: int = dim
        self.prob_name: str = prob  # "categorical", "normal", "uniform"
        # define probability distribution
        klass: Any = object
        if prob == "normal":
            klass = dist.normal.Normal(kwargs["mu"], kwargs["var"])#传入输入字典kwargs的mu键值和var键值
        elif prob == "uniform":
            klass = dist.uniform.Uniform(kwargs["min"], kwargs["max"])
        elif prob == "categorical":
            klass = Categorical(kwargs["k"])
            self.cdim = dim * kwargs["k"]#mnist中cdim=10，dim=1
        self.prob: dist.Distribution = klass#是一个分布对象
        self.params = kwargs#是name,kind,prob,dim键后的字典
    def __str__(self):
        return f"<LatentVariable(name: {self.name}, kind: {self.kind}, prob: {self.prob_name}, dim: {self.dim})>"
    #对象字符串化时的输出，输出对象的几个属性
    def __repr__(self):
        return str(self)

def build_latent_variables(lv_configs) -> Dict[str, LatentVariable]:
    #参数是含有配置文件yaml的latent_variables键的字典,返回的是字典，值为LatnVariable对象
    #主要通过判断kind是类型‘z’还是‘c’，z只能有一个
    #可以简化，把判断z唯一性和判断name重复性的代码删除
    lvars: OrderedDict[str, LatentVariable] = OrderedDict()
    # first of all, add z variable
    count = 0
    for c in lv_configs:
        if c["kind"] == "z":
            if count > 1:
                raise Exception("More than two latent variables of kind 'z' exist!")
            lvars[c["name"]] = LatentVariable(**c)
            #键为c["name"],值为LatenVariable(**c),其中**c是一个字典
            count += 1
    if count == 0:
        raise Exception("Latent variable of kind 'z' doesn't exist!")
    # after that, add other variables
    for c in lv_configs:
        if c["kind"] == "z":
            continue
        if c["name"] in lvars:
            raise Exception("Latent variable name is not unique.")
        lvars[c["name"]] = LatentVariable(**c)
    return lvars#一个字典且里面的值也是字典

class Categorical:
#数据为分类型数据，构造函数传入k(这里为数字的种类10)
    def __init__(self, k: int):
        from torch.distributions.categorical import Categorical as _Categorical
        self.k = k
        p = torch.empty(k).fill_(1 / k)#p的值为k个1/k
        self.prob = _Categorical(p)#一个分布对象
    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        b, c = tuple(x.shape)
        _x = torch.unsqueeze(x, 2)
        oh = torch.empty(b, c, self.k).zero_()
        oh.scatter_(2, _x, 1)
        oh = oh.view(b, c * self.k)
        return oh
    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        x = self.prob.sample(shape)#分布取样，得到形状如shape的prob分布
        x = self.one_hot(x)
        return x

if __name__ == "__main__":
    x = Categorical(10)#代表10个分类
    y1 = x.sample([2,1])#y.shape=(2,10),代表[2,1]转化为[2,10],即10个1-hot编码
    print(y1.shape)
    print(y1)
    y2 = x.sample([5,1])#y.shape=(5,10),代表有100个1-hot编码数
    print(y2.shape)
    print(y2)
