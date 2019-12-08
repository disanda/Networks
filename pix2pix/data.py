###合并原来的data和dataset

from os.path import join
import numpy as np


#参数1为根路径，参数2为目的文件夹
def get_training_set(root_dir, direction):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir, direction)


def get_test_set(root_dir, direction):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir, direction)



from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


#以下三个函数来自utils.py的针对原基本图片的操作
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img
#加载图片后完成到加载为256*256*3的转换
#r, g, b = img.split()  分离三通道
# pic = Image.merge('RGB', (r, g, b))  # 合并三通道


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))



class DatasetFromFolder(data.Dataset):
#参数1为训练图像路径，参数2为输出路径
    def __init__(self, image_dir, direction):
        super().__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenamesA = [x for x in listdir(self.a_path) if is_image_file(x)]
        self.image_filenamesB = [x for x in listdir(self.b_path) if is_image_file(x)]
        #遍历a_path下的文件名后缀

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        #完成了一波PIL转Tensor，并正则化的操作

    def __getitem__(self, index):
    	#该方法可以通过下标A[index]取值 
        a = Image.open(join(self.a_path, self.image_filenamesA[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenamesB[index])).convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
        #两个随机数

        a = a[:,h_offset:h_offset + 256,w_offset:w_offset + 256]
        b = b[:,h_offset:h_offset + 256,w_offset:w_offset + 256]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)#前一个元组是mean，后一个是variance
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenamesA)