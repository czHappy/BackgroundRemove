import cv2
import os
import numpy as np
import random
import math
from tqdm import tqdm

ori_video = ".original_frame"
ori_mask = ".mask_frame"
bg_video = ".background_frame"
dst_path = "result.mp4"
fps = 30


def video_merger(ori_video:str, ori_mask:str, bg_video:str, fps:int, dst_path:str):
    # List of frames' name
    videoframe_name_list = os.listdir(ori_video)
    mask_name_list = os.listdir(ori_mask)
    bg_name_list = os.listdir(bg_video)

    #Loop background video
    ori_len = min(len(videoframe_name_list), len(mask_name_list))
    if len(bg_name_list) < ori_len:
        bg_name_list  *= math.ceil(ori_len / len(bg_name_list))
    assert len(bg_name_list) >= ori_len

    #Get resize ratio
    img = cv2.imread(os.path.join(ori_video, videoframe_name_list[0]))
    bg = cv2.imread(os.path.join(bg_video, bg_name_list[0]))
    mask = cv2.imread(os.path.join(ori_mask, mask_name_list[0]))
    assert img.shape == mask.shape

    rows_original, cols_original, channels_original = img.shape
    rows_bg, cols_bg, channels_bg = bg.shape

    rows_cols_ratio = rows_original / cols_original
    case = 0
    if rows_bg >= rows_original and cols_bg >= cols_original:
        #Only in this case change the scale of background video, in other case we change the scale of original video
        ratio = max(rows_original / rows_bg, cols_original / cols_bg)
        case = 1
    elif rows_bg >= rows_original and cols_bg < cols_original:
        ratio = cols_bg / cols_original
        case = 2
    elif rows_bg < rows_original and cols_bg >= cols_original:
        ratio = rows_bg / rows_original
        case = 3
    elif rows_bg < rows_original and cols_bg < cols_original:
        ratio = min(rows_bg / rows_original, cols_bg / cols_original)
        case = 4
    
    if case == 1:
        dst_scale = (math.ceil(bg.shape[1]*ratio), math.ceil(bg.shape[0]*ratio))
        cols_bg, rows_bg = dst_scale
    else:
        dst_scale = (math.ceil(img.shape[1]*ratio), math.ceil(img.shape[0]*ratio))
        cols_original, rows_original = dst_scale

    #Merge
    size = (cols_original, rows_original)
    fps = fps
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    videoWrite = cv2.VideoWriter(dst_path,fourcc,fps,size)

    for i in tqdm(range(ori_len)):
        img = cv2.imread(os.path.join(ori_video, videoframe_name_list[i]))
        mask = cv2.imread(os.path.join(ori_mask, mask_name_list[i]))
        bg = cv2.imread(os.path.join(bg_video, bg_name_list[i]))

        #Resize background or original video
        if case == 1:
            bg = cv2.resize(bg, dst_scale, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, dst_scale, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (mask.shape[1]*ratio, mask.shape[0]*ratio), interpolation=cv2.INTER_AREA)
        
        if i == 1:
            assert img.shape[0] <= bg.shape[0]
            assert img.shape[1] <= bg.shape[1]
        #Cut background
        bg = bg[int((rows_bg - rows_original) / 2):int((rows_bg + rows_original) / 2), int((cols_bg - cols_original) / 2):int((cols_bg + cols_original) / 2), :]


        f = mask / 255
        result_frame = np.uint8((1 - f) * bg + f * img)
        videoWrite.write(result_frame)
    
    return 

if __name__ == "__main__":
    video_merger(ori_video, ori_mask, bg_video, fps, dst_path)