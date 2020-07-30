# 屏幕坐标中向右X坐标增加，向下是Y轴增加 左上角为原点

import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 已知该图片分辨率512X288 宽度512像素 高度288像素
PATH = './image/1_shrink.png'
cv_img = cv.imread(PATH)
print('cv读取图片返回的格式：', type(cv_img))
print('cv读取图片后的shape：', cv_img.shape)
#cv.imshow('cv_img', cv_img)
#cv.waitKey(0)

plt_img = plt.imread(PATH)
print('matplotlib读取图片返回的格式：', type(plt_img)) #matplotlib返回的不是一个ndarray格式的图片，要转换
print('matplotlib读取图片后的shape：', plt_img.shape)
#plt.imshow(plt_img)
#plt.show()
PIL_img = Image.open(PATH)
print('PIL读取图片返回的格式：', type(PIL_img))

PIL_img = np.array(PIL_img)
print('PIL转换ndarray：', type(PIL_img))
print('PIL读取图片后的shape：', PIL_img.shape)


'''
mean = np.mean(cv_img)
std = np.std(cv_img)
print('*'*80)
print(mean)
print('*'*80)
print(std)
print('*'*80)
recover_plt = plt_img * std + mean
print(recover_plt)
print('*'*80)
print(cv_img - recover_plt)
'''

# 查阅：图像是以矩阵的形式保存的，先行后列。
# 所以，一张 宽×高×颜色通道＝480×256×3 的图片会保存在一个 256×480×3 的三维张量中
# 与实验结果相同，即保存的ndarray格式是HWC

# 图像裁剪
# 一定要注意opencv的图片裁剪参数顺序是不一样的
cv_cut = cv.resize(cv_img, dsize=(500, 250)) #先传入宽，再传入高
print(cv_cut.shape) #(288, 512, 3) -> (250, 500, 3)


# 读取通道顺序问题，opencv读取的通道顺序是BGR 其他是RGB 一定注意
print((cv_img == plt_img).all())
cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

print((PIL_img == cv_img).all())

print((cv_img == plt_img).all()) #为什么这里明明两种编码方式都是RGB，图像矩阵却不相同呢？

print('plt img: \n ', plt_img) #可见plt对输入图片自动做了归一化处理，而cv2和PIL保留原值

# 可视化cv读取的图片的各个通道和其他信息
cv_img = cv.imread(PATH)
print((PIL_img == cv_img).all())

print('cvimg_shape', cv_img.shape)     #读取数据的形状
print('cvimg_size', cv_img.size)       #读取数据的大小
print('cvimg_dtype', cv_img.dtype)     #读取数据的编码格式
#print('cvimg',cv_img)                 #打印img数据
print('cvimg_type', type(cv_img))      #读取img的数据类型
#print(PIL_img)
b, g, r = cv.split(cv_img)    # 通道分离
print('单通道图像的维数', r.ndim)
# print(img.shape[:2])
zeros = np.zeros(cv_img.shape[:2], dtype='uint8')
merge_r = cv.merge([r, zeros, zeros]) # 通道合并
merge_g = cv.merge([zeros, g, zeros])
merge_b = cv.merge([zeros, zeros, b])
print(merge_r)
print(merge_g)
print(merge_b)
plt.subplot(231);plt.imshow(b)
plt.subplot(232);plt.imshow(g)
plt.subplot(233);plt.imshow(r)
plt.subplot(234);plt.imshow(merge_r)
plt.subplot(235);plt.imshow(merge_g)
plt.subplot(236);plt.imshow(merge_b)
plt.show()

merge_all = merge_b + merge_g + merge_r
print(merge_all) # 还原到RGB模式，若用CVimshow则异常，因为它是按照BGR显示
cv.imshow('CV_merge_all', merge_all)   #显示图片
cv.waitKey(0)
merge_all = merge_all[:,:,::-1] #BGR -> RGB
cv.imshow('merge_all', merge_all)   #正常显示图片
cv.waitKey(0)
cv.destroyAllWindows()