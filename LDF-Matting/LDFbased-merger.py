"""
Overall framework for video matting

Input: original video, background video
Output: merged video
Mask predictor:
Merger:

Function flow:
1. Obtaining frames of the original video
2. Predicting masks for each frames
3. Obtaining frames of the background video
4. Merging frames and writing back to video
"""

#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import dataset
from torch.utils.data import DataLoader
from net import LDF

import time
from vf_convert import video2frame, frame2video
from ori_video_merge import video_merger

class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=Path, snapshot='./out/model-40', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
                out  = out2
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = '../eval/maps/ori_frame_temp/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))


if __name__=='__main__':
    start_time1 = time.time()
    print("Cutting frames for original video...")
    ori_fps, ori_frameCount, ori_size, ori_frame_path = video2frame('../input_video/' + sys.argv[1], '../data/ori_frame_temp/image')
    print("Frame saved in " + ori_frame_path)
    print("Cutting frames for background video...")
    bg_fps, bg_frameCount, bg_size, bg_frame_path = video2frame('../input_video/' + sys.argv[2], '../data/bg_frame_temp')
    print("Frame saved in " + bg_frame_path)

    print("Writing txt...")
    ori_frameCount = len(os.listdir(ori_frame_path))
    ftest = open(ori_frame_path + "/../test.txt", 'a')
    for i in range(int(ori_frameCount)):
        ftest.write(str(i+1) + '\n')
    ftest.close()
    print("txt done.")
    interval1 = time.time() - start_time1
    print("Cutting frames using time " + str(interval1))

    print("----------------------------------")

    start_time2 = time.time()
    print("Going into LDF...")
    t = Test(dataset, LDF, ori_frame_path + '/../')
    t.save()
    ori_mask_path = '../eval/maps/ori_frame_temp'
    print("LDF prediction done, saved in " + ori_mask_path)
    interval2 = time.time() - start_time2
    print("LDF prediction using time " + str(interval2))

    print("----------------------------------")

    start_time3 = time.time()
    print("Merging new frames...")
    video_merger(ori_frame_path, ori_mask_path, bg_frame_path, min(ori_fps, bg_fps), '../output_video/' + sys.argv[3])
    print("Merged video saved in " + '../output_video/' + sys.argv[3])
    interval3 = time.time() - start_time3
    print("Merging video using time " + str(interval3))

    print("----------------------------------")

    print("Deleting temporary images...")
    for i in range(len(os.listdir(ori_frame_path))):
        os.remove(ori_frame_path + '/' + str(i+1) + '.jpg')
    os.remove(ori_frame_path + '/../test.txt')
    print("Original video temp frames released.")
    for i in range(len(os.listdir(bg_frame_path))):
        os.remove(bg_frame_path + '/' + str(i+1) + '.jpg')
    print("Background video temp frames released.")
    for i in range(len(os.listdir(ori_mask_path))):
        os.remove(ori_mask_path + '/' + str(i+1) + '.png')
    print("Predicted mask result frames released.")

    print("Working finished.")


