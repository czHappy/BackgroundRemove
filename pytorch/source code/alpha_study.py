import cv2 as cv
import numpy as np
import  matplotlib.pyplot  as plt

def test_where_is_alpha():
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

# test_where_is_alpha() #可以知道alpha通道在第4通道


f = cv.imread('./dataset/labels/1803151818-00000007.png', cv.IMREAD_UNCHANGED)
#cv.imshow('f_4', f) #4通道图片显示有点奇怪


b = cv.imread(r'./image_processing\bg.jpg')

#cv.imshow('b', b) #正常
b = cv.resize(b, (f.shape[1], f.shape[0]), interpolation=cv.INTER_CUBIC)



alpha = f[:,:,-1].astype(float) / 255.0
#alpha = np.where(alpha > 0.2, 1, 0).astype(np.uint8)
cv.imshow('alpha_01', alpha) #正常

# cv.imwrite('alpha1.png', alpha) # 一片漆黑
# cv.imwrite('alpha2.png', alpha.astype(np.uint8)) # 一片漆黑

alpha = alpha[..., np.newaxis]


f = cv.imread('./dataset/imgs/1803151818-00000007.jpg')

#cv.imshow('f_3', f) #正常


fg = np.multiply(alpha, f)

cv.imshow('fg', fg.astype(np.uint8))

bg = np.multiply(1-alpha, b)
cv.imshow('bg', bg.astype(np.uint8))
fusion = fg + bg
fusion = fusion.astype(np.uint8)
print(fusion)
fusion = np.clip(fusion, 0, 255)
cv.imshow('fusion', fusion)
cv.waitKey(0)


'''
m = np.ones([3,3,3])
n = np.ones([3,3]) / 2.0
print(n)
print('*'*80)
print(np.multiply(m,n))
print('*'*80)
print(np.multiply(n,m))

'''