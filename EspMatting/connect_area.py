import cv2
import numpy as np


# mask_sel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, mask_sel = cv2.threshold(mask_sel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# cv2.imshow("mask_sel", mask_sel)
# contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# # 找到最大区域并填充
# area = []
# for j in range(len(contours)):
#     area.append(cv2.contourArea(contours[j]))
# max_idx = np.argmax(area)
# max_area = cv2.contourArea(contours[max_idx])
#
# cv2.fillConvexPoly(mask_sel, contours[max_idx], 0)
# # cv2.fillPoly(mask_sel, [contours[max_idx]], 0)
# cv2.imshow("image", img) # 显示图片
# cv2.imshow("connect", mask_sel)
# cv2.waitKey(0) #等待按键


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 找到最大区域并填充
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

img = cv2.imread('test\sr\matting_person7.jpg')
mask_sel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask_sel = cv2.threshold(mask_sel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("mask_sel", mask_sel)
res = find_max_region(mask_sel)
cv2.imshow("res", res)
cv2.waitKey(0) #等待按键

