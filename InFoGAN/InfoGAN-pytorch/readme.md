1.配置文件：mnist.yaml

2.数据集: dataset.py
>图片大小为64*64

3.测试:infer.py

4.通用工具:util.py
  - 加载配置文件
  - 设备设置
  - 生成图像(包括随机，离散型c，连续c)

5.输入类:variable.py
  - 定义输入类:LatentVariable
  - 创造输入对象字典的函数:build_latent_variables()
  - 分类one-hot转换类:Categorical

6.损失类:loss.py
 - 对抗训练损失(Adv):AdversariaLoss
  a.用交叉熵BCE,判断输入和标签的近似的程度
  b.真实图片输入到Dis,Dis输出到DHead，DHead的输出连同Label(真实图片为1,随机数为0)给对抗训练损失Adv,
  c.其输出一个值(代表图片经过Did和DHead后和label的近似程度)
 
 
 - 信息判别器(info):InfoGANLoss
  >Dis输出到QHead，QHead输出给InFo作为第一个参数，第二个参数来自G输入的规则变量(包括离散变量和连续变量)
  >输出第一个值为全部变量损失总和，第二个值为各个变量的loss
  a.离散型变量，CrossEntropyLoss()
  b.连续型变量，NormarNLLLoss()
  
 - 连续型变量判别类:NormalNLLLoss
 
 7.模型类:model.py
  - Noise(让输入x加上一个标准正态分布的随机数)
  - Generator(带规则的离散变量:cat和连续变量:c1-c3的随机数作为输入，输出为图片[-1,1,64,64])
  - Discriminator(输入图片:[-1,1,64,64]，输出[-1,512,4,4])
  - DHead(判断图片真假的模型，输入为Dis的输出，输出[-1,1,4,4])
  - QHead(判断图片类型的模型，输入为Dis的输出, 输出一个字典,分类变量值为[-1,k],连续变量值为[-1,2])
 
 8.trainer.py
  - train的核心,接train的配置
 
 9.logger.py
 - 重写了logging模块,记录时间、各loss损失、迭代次数,保存模型参数于models
 - 载入tensorboard，记录上述对应于run文件夹
 
 10.train.py
 创造trainner的配置，作为参数传入trainer
     -第一个参数为dataloader对象，数据集载入
     -第二个参数latent_vars:输入数据对象latentvarible组成的字典
     -第三个参数models:模型对象组成的字典
     -第四个参数opts:优化器对象组成的字典
     -第五个参数losses:损失函数对象组成的字典
     -第六个参数configs["traning"]:关于训练设置参数的字典
     -第七个参数logger:自定义的Logger类

  
