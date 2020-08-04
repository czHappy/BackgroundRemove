import cv2 as cv
import os, sys
SAVE_PATH = r"./dataset/masks"
def generate_masks_from_matting(basedir=r"./dataset/labels"):
    imgs = os.listdir(basedir) #取出文件名字
    for img in imgs:
        path = os.path.join(basedir, img)
        m = cv.imread(path, cv.IMREAD_UNCHANGED)
        # Otsu 滤波
        ret, mask = cv.threshold(m[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        print(os.path.join(basedir, 'masks', img))
        cv.imwrite(os.path.join(SAVE_PATH, img), mask)



generate_masks_from_matting()
