import functools
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import util
from variable import LatentVariable

LABEL_REAL, LABEL_FAKE = 1, 0

class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCELoss(reduction="mean")
        self.device = util.current_device()
    def __call__(self, y_hat: torch.Tensor, label: int):
    #象函数一样调用对象,y_hat:是qhead的输出:[-1,1,4,4]
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")
        y_true = torch.full(y_hat.size(), label, device=self.device)#y_true根据传入的第二个参数label,让元素全部至0或1
        return self.loss(y_hat, y_true)

class InfoGANLoss:
#对比多个信息
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        self.latent_vars = latent_vars
        self.discrete_loss = nn.CrossEntropyLoss()#离散型
        self.continuous_loss = NormalNLLLoss()#连续型
        self.device = util.current_device()
    def __call__(self, cs_hat: Dict[str, torch.Tensor], cs_true: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#函数化调用对象时，输入两个字典，第一个cs_hat是随机输入条件产生的假图像通过D和Q获得的条件，c_true是随机输入的条件
        if cs_hat.keys() != cs_true.keys():
            raise Exception("The keys of cs_hat is different from cs_true")
        losses: List[torch.Tensor] = []
        details: Dict[str, torch.Tensor] = {}
        for key in cs_hat.keys():
            c_hat, c_true = cs_hat[key], cs_true[key]#c_hat,c_true是具体键下的值，包括c1,c2,c3
            if self.latent_vars[key].prob_name == "categorical":
                # loss for discrete variable
                _, targets = c_true.max(dim=1)#max(dim=1)返回元素属性中最大值的索引，刚好就是categorical
                loss = self.discrete_loss(c_hat, targets)
            elif self.latent_vars[key].prob_name == "normal":
                # loss for continuous variable
                dim: int = self.latent_vars[key].dim#为1
                mean, ln_var = c_hat[:, :dim], c_hat[:, dim:]#c_hat刚好有两个值[-1,2],一个给mean,一个给ln_var
                loss = self.continuous_loss(c_true, mean, ln_var)
            loss = loss * self.latent_vars[key].params["weight"]
            details[key] = loss
            losses.append(loss)
        return functools.reduce(lambda x, y: x + y, losses), details
        #其中的reduce()是求losses列表中元素的合,返回第一个值是总和，第二个是各个变量loss的字典
class NormalNLLLoss:
    def __call__(self, x: torch.Tensor, mean: torch.Tensor, ln_var: torch.Tensor) -> torch.Tensor:
        x_prec = torch.exp(-ln_var)#x是随机值
        x_diff = x - mean
        x_power = (x_diff * x_diff) * x_prec * -0.5
        loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
        return torch.mean(loss)
