import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from model import segnet
INPUT_SIZE = 1024
device = 'cuda'
model_path = "pre_trained/attunet/model/model_obj.pth"
video_path = "video_demo/meeting_room_hard.mp4"

def load_model(model_path):
    print('Loading model from {}...'.format(model_path))
    if device == 'cpu':
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/5
        # myModel = torch.load(args.model, map_location=lambda storage, loc: storage) #把GPU上训练的模型加载到CPU上
        myModel = segnet.SegMattingNet()
        myModel.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage)['state_dict'])

    else:
        # myModel = segnet.SegMattingNet() 用字典的话取消注视此两行
        # myModel.load_state_dict(torch.load(model_path)['state_dict'])
        myModel = torch.load(model_path)  # 用整个模型

    myModel.eval() # 设置模型为eval模式
    myModel.to(device)

    return myModel

def matting(net, video, result, fps=30):
    # video capture
    vc = cv2.VideoCapture(video)
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    if not rval:
        print('Failed to read the video: {0}'.format(video))
        exit()

    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    h, w = frame.shape[:2]
    rh = INPUT_SIZE
    rw = INPUT_SIZE

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))

    print('Start matting...')
    with tqdm(range(int(num_frame)))as t:
        for c in t:
            origin_img = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_CUBIC)
            frame_np = (origin_img - (121.01928, 121.11276, 122.93247)) / 255.0  # 归一化
            tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)  # 变单张图片为tensor
            # torch.FloatTensor 是torch.Tensor的简称
            tensor_4D[0, :, :, :] = torch.FloatTensor(frame_np.transpose(2, 0, 1))  # HWC=>CHW
            inputs = tensor_4D.to(device)
            with torch.no_grad():
                seg, alpha = net(inputs)
            matte_tensor = alpha.repeat(1, 3, 1, 1) # NCHW
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0) #HWC
            view_np = matte_np * origin_img + (1 - matte_np) * np.full(origin_img.shape, 255.0)
            view_np = view_np.astype(np.uint8)
            view_np = cv2.resize(view_np, (w, h))
            video_writer.write(view_np)
            rval, frame = vc.read()
            c += 1
    video_writer.release()
    print('Save the result video to {0}'.format(result))

if __name__ == '__main__':
    result = os.path.splitext(video_path)[0] + '_{0}.mp4'.format('result_espnew1080')
    myModel = load_model(model_path)
    matting(myModel, video_path, result, fps=30)
