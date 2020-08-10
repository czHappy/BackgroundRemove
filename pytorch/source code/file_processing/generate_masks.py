import cv2 as cv
import os, sys
import numpy as np
SAVE_PATH = r"../dataset/masks"
def generate_masks_from_matting(basedir=r"../dataset/labels"):
    imgs = os.listdir(basedir) #取出文件名字
    for img in imgs:
        path = os.path.join(basedir, img)
        m = cv.imread(path, cv.IMREAD_UNCHANGED)
        # Otsu 滤波
        ret, mask = cv.threshold(m[:, :, 3], 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU) # 二值化
        print(os.path.join(basedir, 'masks', img))
        cv.imwrite(os.path.join(SAVE_PATH, img), mask)

def generate_alpha_from_RGBA(basedir=r"./dataset/labels"):
    imgs = os.listdir(basedir)  # 取出文件名字
    for img in imgs:
        path = os.path.join(basedir, img)
        m = cv.imread(path, cv.IMREAD_UNCHANGED)
        m = m[:, :, 3]
        print(os.path.join(basedir, 'masks', img))
        cv.imwrite(os.path.join(SAVE_PATH, img), m)

np.set_printoptions(threshold=sys.maxsize)

generate_masks_from_matting()
img = cv.imread(os.path.join(SAVE_PATH, '1803151818-00000003.png'), cv.IMREAD_GRAYSCALE)
print('origin:', img)
cv.imshow('origin', img)
cv.waitKey(0)
# generate_alpha_from_RGBA()
'''
# test read
np.set_printoptions(threshold=sys.maxsize)
img = cv.imread(os.path.join(SAVE_PATH, '1803151818-00000003.png'), cv.IMREAD_GRAYSCALE)
print('origin:', img)
# img = np.where(img > 0, 255, 0).astype(np.uint8)
img = img.astype(float) / 255.0
print(img.shape)
print(img)
print(  (img == np.zeros_like(img)).all() )

# 尝试保存浮点数像素
cv.imwrite('./float_image.png', img.astype(np.uint8))
# 读取浮点数像素图像
img_float = cv.imread('./float_image.png')
cv.imshow('float', img_float)

print('float:', img_float) # 保存不了，显示的是自动取整了

cv.imshow('float2uint8', img_float.astype(np.uint8))


cv.imshow('alpha', img)
cv.waitKey(0)

'''