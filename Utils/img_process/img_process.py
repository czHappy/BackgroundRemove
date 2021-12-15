# 图像resize
def img_resize():
    img = cv2.imread("./demo_video/gy.jpg")
    print(img.shape)
    # (W,H)
    img = cv2.resize(img, (1920, 1080),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("./demo_video/gy3.jpg", img)
    print(img.shape)

# 图像转tensor
def np2Tensor(array):
    ts = (2, 0, 1) # PIL H x W x C np是H x W x C  torch.tensor C x H x W
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))
    return tensor

# img = np2Tensor(img).unsqueeze(0) #变成4维度张量 添加batch维度

# video 转 img sequence
def mp42imgs(path,imgs_path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))
    for i in range(int(frames)):
        print(i)
        ret, frame = videoCapture.read()
        cv2.imwrite(os.path.join(imgs_path,"%5d.png" % (i)), frame)

# img sequence 转 video
import cv2
import os
import numpy as np
def imgs2mp4():
    base = r'./images/gy'
    img = cv2.imread(os.path.join(base, '00000.jpg'))  #读取第一张图片
    imgInfo = img.shape
    size = (imgInfo[1],imgInfo[0])  #获取图片宽高度信息
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    videoWrite = cv2.VideoWriter(os.path.join("./results",'output.mp4'),fourcc,25,size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，25帧，视频大小（图片大小））
    imgs = os.listdir(base)
    for f in imgs:
        print(f)
        fileName = os.path.join(base, f)    #循环读取所有的图片
        img = cv2.imread(fileName)
        print(img.shape)
        videoWrite.write(img)# 将图片写入所创建的视频对象
    #videoWrite.release()  # 释放
    print('end!')


# 视频网络处理和保存
# video视频路径 result结果保存路径 net使用的网络 fps保存结果的fps
def video_seg(video, result, net, fps=30):

    videoCapture = cv2.VideoCapture(video) #参数0表示打开内置摄像头

    num_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)) #获取视频FPS
    ret, pre_frame = videoCapture.read() #读下一帧

    # pre_frame = cv2.flip(pre_frame, 1) # 水平翻转
    pre_seg = get_alpha(pre_frame, net) 

    ret, cur_frame = videoCapture.read()
    cur_frame = cv2.flip(cur_frame, 1)
    cur_seg = get_alpha(pre_frame, net)

    next_frame = None
    next_seg = None

    h, w = pre_frame.shape[:2]
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))# VideoWriter

    for t in tqdm(range(num_frame - 1)):
        ret, next_frame = videoCapture.read()
        if not ret:
            break

        next_seg = get_alpha(next_frame, net)
        res_seg = ofd(pre_seg, cur_seg, next_seg)
        # res_seg = cur_seg
        res = get_out(cur_frame, res_seg)

        pre_seg = cur_seg
        cur_seg = next_seg
        cur_frame = next_frame

        # 存储当前帧到VideoWriter
        video_writer.write(res)
    videoCapture.release() #释放videoCapture
    print('Save the result video to {0}'.format(result))