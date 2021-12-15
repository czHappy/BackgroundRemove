import cv2
import numpy as np
img = np.zeros((512,512, 3), dtype=np.uint8)#random.random()方法后面不能加数据类型
#img = np.random.random((3,3)) #生成随机数都是小数无法转化颜色,无法调用cv2.cvtColor函数

cv2.imwrite('./black.png', img)

