import cv2 as cv
import os
import numpy as np
import shutil
# usage 排列组合与背景替换
root_path = r'C:\Users\cz\Desktop\dataset\test'

# 存图目标文件
img_path = r'C:/Users/cz/Desktop/dataset/test/new_imgs'
alpha_path = r'C:/Users/cz/Desktop/dataset/test/new_alphas'
# mask_path = r''

bg_path = r'C:\Users\cz\Desktop\dataset\test\bgs'
train_file = r'C:\Users\cz\Desktop\dataset\test\train.txt'
# 流程
# 30张背景图
# 从原数据里取出 400张 均匀从各个视频帧里取出来
# 全部组合 一共增广12000张图片
# 背景替换策略是，现将背景resize到数据集图片大小 然后使用背景替换公式做融合

def sample(train_file):
    bg_list = os.listdir(bg_path)
    print(bg_list)
    file = open(train_file)
    cnt = -1
    for line in file.readlines():
        print(line)
        cnt = cnt + 1
        if cnt % 2 != 0:
            continue
        line = line.strip('\n').split()
        img = os.path.join(root_path, line[0])

        img = cv.imread(img)

        alpha = os.path.join(root_path, line[2])

        alpha = cv.imread(alpha)
        alpha = alpha.astype(np.float16) / 255.0

        #w,h = img.shape
        print(img.shape)
        # exit(1)
        bg_cnt = -1
        for bg in bg_list:  # 30张bg图片
            bg_cnt = bg_cnt + 1
            bg_img = cv.imread(os.path.join(bg_path, bg))
            bg_img = cv.resize(bg_img, (img.shape[1], img.shape[0]),interpolation=cv.INTER_CUBIC)
            result = alpha * img + (1 - alpha) * bg_img
            result[result < 0] = 0
            result[result > 255] = 255
            result = result.astype(np.uint8)
            # 保存结果
            cv.imwrite(os.path.join(img_path, 'new_{}_{}.png'.format(cnt, bg_cnt)), result)
            shutil.copyfile(os.path.join(root_path, line[2]), os.path.join(alpha_path, 'new_{}_{}.png'.format(cnt, bg_cnt)))




sample(train_file)