import numpy as np
import time
import cv2
eps = 1e-6

#调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    start = time.time()
    #从摄像头读取图片
    sucess, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow("img", img)
    end = time.time()
    print("fps = ", 1.0 / (end - start + eps))
    #转为灰度图片
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #显示摄像头，背景是灰度。
    #cv2.imshow("img",gray)
    #保持画面的持续。
    k = cv2.waitKey(1)
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k==ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg",img)
        cv2.destroyAllWindows()
        break
#关闭摄像头
cap.release()

# 试验说明本摄像头可以支持到30fps以上 对1280X720