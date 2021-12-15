'''
Author  : Zhengwei Li
Version : 1.0.0 
'''

import cv2
import os
import random as r
import numpy as np

import torch
import torch.utils.data as data


# ============================================================================================================
def crop_patch_augment(_img, _mask, _alpha, patch):


    (h, w, c) = _img.shape
    scale = 0.75 + 0.5*r.random()

    _img = cv2.resize(_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    _mask = cv2.resize(_mask, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    _alpha = cv2.resize(_alpha, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    (h, w, c) = _img.shape

    if r.random() < 0.5:
        if h>patch and w>patch:
            x = r.randrange(0, (w - patch))
            y = r.randrange(0, (h - patch))
            

            _img = _img[y:y + patch, x:x + patch, :]
            _mask = _mask[y:y + patch, x:x + patch, :]
            _alpha = _alpha[y:y + patch, x:x + patch, :]
        else:

            _img = cv2.resize(_img, (patch,patch), interpolation=cv2.INTER_CUBIC)
            _mask = cv2.resize(_mask, (patch,patch), interpolation=cv2.INTER_NEAREST)
            _alpha = cv2.resize(_alpha, (patch,patch), interpolation=cv2.INTER_CUBIC)
    else:

        _img = cv2.resize(_img, (patch,patch), interpolation=cv2.INTER_CUBIC)
        _mask = cv2.resize(_mask, (patch,patch), interpolation=cv2.INTER_NEAREST)
        _alpha = cv2.resize(_alpha, (patch,patch), interpolation=cv2.INTER_CUBIC)


    # flip
    if r.random() < 0.5:
        _img = cv2.flip(_img,0)
        _mask = cv2.flip(_mask,0)
        _alpha = cv2.flip(_alpha,0)

    if r.random() < 0.5:
        _img = cv2.flip(_img,1)
        _mask = cv2.flip(_mask,1)
        _alpha = cv2.flip(_alpha,1)

    return _img, _mask, _alpha

# 图片背景和肖像增强
def im_bg_augment(_img, _mask):

    if r.random() < 0.2:
        _img_portrait = np.multiply(_mask, _img) #取出肖像
        _img_bg = np.multiply(1 - _mask, _img) #取出背景
        # 背景三个通道做一些随机变换
        _img_bg[:,:,0] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,0])
        _img_bg[:,:,1] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,1])
        _img_bg[:,:,2] = np.multiply(np.random.rand()+0.2, _img_bg[:,:,2])

        _img_bg[_img_bg>=1.0] = 1.0  #防止超过1
        _img_new = _img_bg + _img_portrait #合成
    else:
        _img_new = _img #80%的概率不做数据增强

    return _img_new


def np2Tensor(array):
    ts = (2, 0, 1) # np是H x W x C PIL H x W x C torch.tensor C x H x W
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))    
    return tensor
"""
    dataset: human_matting 
"""

# 制作自己的Dataset,要实现 __init__ __getitem__ __len__方法
# human_matting被getattr调用
class human_matting(data.Dataset):

    def __init__(self, base_dir, imglist, patch):

        super().__init__()
        self._base_dir = base_dir
        # 相当于把 '/home/lzw/Disk/data/semantic_Segmentation/human_matting/' 和 'train.txt' 相拼接起来了
        # 然后将每一行读取出来，每一行是原图+mask的路径，存入file_list成员变量
        with open(os.path.join(self._base_dir, imglist)) as f:
            self.file_list = f.readlines()

        self.file_list = self.file_list # 猜测是为了让file_list 到外层作用域
        self.data_num = len(self.file_list)
        self.patch = patch
        print("Dataset : aifenge, deep automatic potrait matting and supervisely !")
        print('file number %d' % self.data_num)

    # 可以这么说，Mask是Matte的一种特例。在Mask里，只有两种透明度，1和0，即完全透明和完全不透明。
    # Mask的产生是为了去除合成时的锯齿而设计的，但锯齿没了，不过合成痕迹太明显，显得很不真实。
    # 而Matte则可以包含很多层次的透明度，图像中每个像素都可以有自己的透明度，
    # 这些像素的透明度有着丰富的层级，可以合成、融合。

    def __getitem__(self, index):

        img_path, mask_path, alpha_path = self.getFileName(index)
        _img = cv2.imread(img_path).astype(np.float32) #根据绝对路径用cv读取图片成为ndarray
        # bright
        if r.random() < 0.5: # 随机作用
            if r.random() < 0.5: # clip用于将矩阵的值限制在minx, maxx之间
                _img = np.uint8(np.clip(_img + r.randrange(0, 45), 0, 255))
            else:
                _img = np.uint8(np.clip(_img - r.randrange(0, 45), 0, 255))            

        _img = (_img - (115.4225, 121.7716, 128.3485,)) / 255.0 #归一化
        #(115.42250569210887, 121.77165961887842, 128.34856405192102)
        _mask = cv2.imread(mask_path).astype(np.float32) #(0,1) 为什么这里是 (0,1)，是区间还是只有两个值 应该是区间，因为下面有将alpha赋值为mask，而alpha总该是区间....

        # 此处修改
        _alpha = cv2.imread(alpha_path).astype(np.float32)
        _alpha = _alpha / 255.0 #注意归一化

        _img = im_bg_augment(_img, _mask)
        _img, _mask, _alpha = crop_patch_augment(_img, _mask, _alpha, self.patch)

        # numpy 转换为tensor格式， 即HWC -> CHW
        _img = np2Tensor(_img)
        _mask = np2Tensor(_mask)
        _alpha = np2Tensor(_alpha)

        # 扩展维度变成标准4-D维度 不知道这里为什么要扩展 可能是因为dataset被dataloader提取时没有经过transform ToTensor
        _mask = _mask[0,:,:].unsqueeze_(0) #这里把mask的第0个通道，即B通道扩展成4D，此时mask是单通道
        _alpha = _alpha[0,:,:].unsqueeze_(0)

        sample = {'image': _img, 'mask': _mask, 'alpha': _alpha}

        return sample

    def __len__(self):
        return self.data_num
    
    # 此处修改 origin+mask+alpha
    '''
    def getFileName(self, idx):

        line = self.file_list[idx] #取出第idx个样本
        line = line.replace(' ', '\t')
        name = line.split('\t')[0]
        nameIm = os.path.join(self._base_dir, name)
        name = line.split('\t')[1].split('\n')[0]
        nameTar = os.path.join(self._base_dir, name)
        #分别找到img 和tarjet的绝对路径
        return nameIm, nameTar
    '''

    def getFileName(self, idx):

        line = self.file_list[idx]  # 取出第idx个样本
        line = line.split(' ')
        img = os.path.join(self._base_dir, line[0])
        mask = os.path.join(self._base_dir, line[1])
        alpha = os.path.join(self._base_dir, line[2].split('\n')[0]) #注意这里要把最后的换行符号消去
        #print(img, mask, alpha, sep='\n')
        # 分别找到img 和tarjet的绝对路径
        return img, mask, alpha

