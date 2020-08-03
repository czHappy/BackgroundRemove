import cv2 as cv
import numpy as np
import  matplotlib.pyplot  as plt
img = cv.imread('./dataset/labels/1803151818-00000007.png')
print(img.shape)
#k = plt.imread('./dataset/labels/1803151818-00000007.png')
#plt.imshow(k)
#plt.show()
img_rgba = cv.imread('./dataset/labels/1803151818-00000007.png', cv.IMREAD_UNCHANGED)
print(img_rgba.shape)

#cv.imshow('rgba',img_rgba)
#cv.waitKey(0)


mask1 = img_rgba[:,:,0]
mask2 = img_rgba[:,:,1]
mask3 = img_rgba[:,:,2]
mask4 = img_rgba[:,:,3]

print(mask1.shape)

#cv.imshow('mask1', mask1)
#cv.waitKey(0)
#cv.imshow('mask2', mask2)
#cv.waitKey(0)
#cv.imshow('mask3', mask3)
#cv.waitKey(0)
cv.imshow('mask4', mask4) #通过测试，可以知道RGBA 4通道图像第四个通道是alpha通道 可以直接取出
cv.waitKey(0)
print(mask4)
cv.imwrite('./alpha.png',mask4)
print('*'*90)
mask_bi = np.where(mask4 > 50, 255, 0).astype(np.uint8)
print(mask_bi)
cv.imwrite('./mask_bi.png', mask_bi)
cv.imshow('mask_bi', mask_bi)
cv.waitKey(0)
