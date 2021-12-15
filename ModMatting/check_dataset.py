import os
import cv2 as cv
root = r'E:\Data_Backup\mobile-torch1.x\dataset'
imgs = os.path.join(root, 'imgs')
alphas = os.path.join(root, 'alphas')
imgs_list = os.listdir(imgs)
alphas_list = os.listdir(alphas)
sorted(imgs_list)
sorted(alphas_list)
assert  len(imgs_list) == len(alphas_list)

for img, alpha in zip(imgs_list, alphas_list):
    img_r = cv.imread(os.path.join(imgs, img))
    alpha_r = cv.imread(os.path.join(alphas, alpha))
    h, w, c = img_r.shape
    rh = 512
    rw = int(w/h*512)
    rh = rh - rh % 32
    rw = rw - rw % 32

    img_r = cv.resize(img_r, (rw, rh))
    alpha_r = cv.resize(alpha_r, (rw, rh))
    print(img_r.shape, "----", alpha_r.shape)

