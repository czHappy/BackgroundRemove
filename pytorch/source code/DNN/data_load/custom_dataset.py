import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2 as cv
from torchvision import transforms

# src_dir = E:\数据备份\Matting_Human_Half\matting_human_half\clip_img\1803151818\clip_00000000
# matting = E:\数据备份\Matting_Human_Half\matting_human_half\matting\1803151818\matting_00000000


class HumanMatting(Dataset):
    def  __init__(self, base_dir, list_file):
        self.image_label_list = self.read_file(base_dir, list_file)
        self.base_dir = base_dir
        self.len = len(self.image_label_list)
        self.toTensor = transforms.ToTensor()

    def read_file(self, base_dir, list_file):
        PATH = os.path.join(base_dir, list_file) #list_file的路径
        img_lb_list = []
        with open(PATH, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # rstrip() 方法用于删除字符串尾部指定的字符，默认字符为所有空字符，包括空格、换行(\n)、制表符(\t)等。
            content = line.rstrip().split(' ')
            name = content[0]
            label = content[1]
            img_lb_list.append((name, label))
        return img_lb_list


    def __getitem__(self, index):
        index = index % self.len # 防止有repeated
        img_name, lb_name = self.image_label_list[index]
        img_path = os.path.join(self.base_dir, img_name)
        lb_path = os.path.join(self.base_dir, lb_name)
        bgr_img = cv.imread(img_path)
        bgr_lb = cv.imread(lb_path) #注意 ，这是BGR通道顺序
        img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        lb = cv.cvtColor(bgr_lb, cv.COLOR_BGR2RGB)
        # 变成tensor
        img = self.toTensor(img) #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        lb = self.toTensor(lb)
        sample = {'image': img, 'label': lb}
        return sample

    def __len__(self):
        return self.len

    def size(self):
        return self.len


# 注意没有对图片预先进行归一化N(0,1) 但是ToTensor做了归一化

def cv_show_image(title, image):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :return:
    '''
    channels = image.shape[-1]
    if channels == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # 将RGB转为BGR 一会实验不转行不行
    cv.imshow(title, image)
    cv.waitKey(0)

if __name__ == "__main__":
    train_list = 'train.txt'
    train_dir = '../../dataset/'
    train_data = HumanMatting(train_dir, train_list)
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
    print(train_data.size())

    for batch_sample in train_loader:
        img = batch_sample['image'][0].numpy() #只选出第0个图片
        lb = batch_sample['label'][0].numpy()
        img = img.transpose(1, 2, 0)
        lb = lb.transpose(1, 2, 0)
        cv_show_image('img', img)
        cv_show_image('lb', lb)


