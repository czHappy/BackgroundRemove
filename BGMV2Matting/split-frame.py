import os
import cv2

cap = cv2.VideoCapture("./video_test/wqf.mp4")

success, img1 = cap.read()
cnt = 0
while success:
    pre = img1
    cv2.imwrite('./frames/bg_{}.png'.format(cnt), pre)
    cnt = cnt+1
    success, img1 = cap.read()

