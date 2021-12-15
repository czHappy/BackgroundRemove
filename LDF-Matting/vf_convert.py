import cv2
import os
import numpy as np
from tqdm import tqdm
import sys
import subprocess


def video2frame(video_name, temp_frame='./temp_frame'):
    video = cv2.VideoCapture(video_name)
    fps = video.get(cv2.CAP_PROP_FPS)
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print(fps, frameCount, size)    # 30.0 960.0 (960,480)

    os.makedirs(temp_frame, exist_ok=True)

    if video.isOpened():
        for c in tqdm(range(int(frameCount))):  # 循环读取视频帧
            rval, frame = video.read()
            if rval:
                cv2.imwrite(temp_frame + '/' + str(c+1) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                cv2.waitKey(1)
            else:
                break
    
    print('frames saved success in ' + str(temp_frame))
    video.release()

    return fps, frameCount, size, temp_frame

def get_fps(video_name1, video_name2):
    video1 = cv2.VideoCapture(video_name1)
    video2 = cv2.VideoCapture(video_name2)
    fps1 = video1.get(cv2.CAP_PROP_FPS)
    fps2 = video2.get(cv2.CAP_PROP_FPS)
    fps = min(fps1, fps2)
    video1.release()
    video2.release()
    return fps


def video2frame_ffmpeg(video_name, temp_frame='./temp_frame', fps=30):
    video = cv2.VideoCapture(video_name)
    fps = int(fps)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    command = "ffmpeg -i {} -f image2 -vf fps={} {}/%d.jpg".format(video_name, fps, temp_frame)
    print(command)
    print(os.getcwd())
    subprocess.call(command, shell=True)
    print('frames saved success in ' + str(temp_frame))
    frameCount = len(os.listdir(temp_frame))
    return fps, frameCount, size, temp_frame


def frame2video(fps, size, temp_frame, output_vname='output.mp4'):
    # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videoWrite = cv2.VideoWriter(output_vname, fourcc, fps, size)

    files = os.listdir(temp_frame)
    out_num = len(files)
    for t in tqdm(range(out_num)):
        frame = cv2.imread(temp_frame + '/' + str(t+1) + '.jpg')
        videoWrite.write(frame)  # 将图片写入所创建的视频对象
        # print(str(i) + ' done')

    print('merged success into ' + str(output_vname))


if __name__ == '__main__':
    # fps, frameCount, size, temp_frame = video2frame('output.mp4')
    # frame2video(fps, size, temp_frame, output_vname='new_output.mp4')
    fps, frameCount, size, temp_frame = video2frame(sys.argv[1])
    frame2video(fps, size, temp_frame, output_vname=sys.argv[2])

