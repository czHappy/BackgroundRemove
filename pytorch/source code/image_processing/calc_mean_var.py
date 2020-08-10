import os
import cv2 as cv
import numpy as np
import sys
from tqdm import tqdm
from time import time

IMG_DIR = '../dataset/imgs'
def calc_mean(img_dir=IMG_DIR):
    print('==> calc mean......')
    img_list = os.listdir(img_dir)
    mean_b = mean_g = mean_r = 0
    for img_name in tqdm(img_list):
        img = cv.imread(os.path.join(IMG_DIR, img_name))
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(img_list)
    mean_g /= len(img_list)
    mean_r /= len(img_list)

    return mean_b, mean_g, mean_r

def calc_var(img_dir=IMG_DIR):
    print('==> calc var......')
    img_list = os.listdir(img_dir)
    mean_b, mean_g, mean_r = calc_mean(img_dir)
    var_b = var_g = var_r = 0
    total = 0
    for img_name in img_list:
        img = cv.imread(os.path.join(IMG_DIR, img_name))
        var_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        var_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        var_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))
        total += np.prod(img[:, :, 0].shape)

    var_b = np.sqrt(var_b / total)
    var_g = np.sqrt(var_g / total)
    var_r = np.sqrt(var_r / total)

    return var_b, var_g, var_r

print(calc_mean()) #多值返回 返回的是元组

print(calc_var())



