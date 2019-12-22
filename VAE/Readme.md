## 训练原理

一个编码器 Q(z|x)

一个解码器 P(y|z)

一个 KL 散度

完事

## 训练过程

1.初始化

初始化函数一般需要考虑到后面的激活函数

2.前向传播

一般有多层，一层中包括卷积函数、激活函数、BN函数。

该层计算出loss，之后会根据loss最小的方向去BP

3.后向传播(BP)

更具loss最小的梯度方向去更新参数w

参考:
https://github.com/wiseodd/generative-models
https://wiseodd.github.io/techblog/2017/01/24/vae-pytorch/
