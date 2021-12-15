import os
import cv2 as cv
import  numpy as np
# usage 从alpha生成mask
def generate_masks_from_matting(SAVE_PATH=r"C:\Users\cz\Desktop\fvc_process\masks", basedir=r"C:\Users\cz\Desktop\fvc_process\alphas"):
    assert os.path.exists(SAVE_PATH)
    imgs = os.listdir(basedir) # 取出文件名字
    idx = 1
    for img in imgs:
        path = os.path.join(basedir, img)
        m = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # Otsu 滤波
        ret, mask = cv.threshold(m, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # print(os.path.join(basedir, 'masks', img))
        cv.imwrite(os.path.join(SAVE_PATH, img), mask)
        if idx % 100 == 0:
            print('{} complete.'.format(idx))
        idx+=1
def generate(basedir, imgs, masks, alphas):
    # basedir 根目录
    # imgs 根目录下的图片文件夹的名称
    # masks 根目录下的mask文件夹的名称
    # alphas 根目录下的alpha文件夹的名称
    img_list = os.listdir(os.path.join(basedir, imgs))
    mask_list = os.listdir(os.path.join(basedir, masks))
    alpha_list = os.listdir(os.path.join(basedir, alphas))
    img_list.sort() #由于同名仅仅后缀不相同的关系 故而sort之后必然保持对应关系
    mask_list.sort()
    alpha_list.sort()
    with open(os.path.join(basedir, 'train.txt'), 'w') as f:
        assert len(img_list) == len(mask_list) and len(mask_list) == len(alpha_list)
        for i in range(len(img_list)):
            item = os.path.join(imgs, img_list[i]) + ' ' + os.path.join(masks, mask_list[i]) + ' ' + os.path.join(alphas, alpha_list[i]) + '\n'
            f.write(item)

generate_masks_from_matting()


# def test(img, mask):
#     img = cv.imread(img)
#     mask = cv.imread(mask)
#     cv.imshow('img', img)
#     cv.waitKey(0)
#
#     cv.imshow('mask', mask)
#     cv.waitKey(0)
#     mask = mask.astype(np.float) / 255.0
#     res = img * mask
#     cv.imshow('res', res)
#     cv.waitKey(0)
# generate_masks_from_matting()
# test(img=r'C:\Users\cz\Desktop\fvc_process\imgs\laptop_a0256_outdoor_0954.jpg', mask=r'C:\Users\cz\Desktop\fvc_process\masks\laptop_a0256_outdoor_0954.png')
