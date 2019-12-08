from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from data import is_image_file, load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='toy')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)


if opt.direction == "b2a":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
#找出里面的图片文件，is_image_file()函数通过后缀名去看

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
#b2a
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
