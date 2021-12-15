import cv2
import torch
import numpy as np
from tqdm import tqdm
from model import segnet
import os
torch.set_grad_enabled(False) # 不要计算导数了在interfere的时候
INPUT_SIZE = 512
EPSILON = 0.2
device = 'cuda'
model_path = "pre_trained/erd/model/ckpt_lastest.pth"
video_path = "./video_demo/wqf.mp4"

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
        myModel = torch.load(model_path) # 用整个模型

    myModel.eval() # 设置模型为eval模式
    myModel.to(device)

    return myModel


def get_alpha(image, net):
    # opencv

    image_resize = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (121.0192816826140, 121.11276462604845, 122.93247702952327)) / 255.0  # 归一化
    tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)  # 变单张图片为tensor
    # torch.FloatTensor 是torch.Tensor的简称
    tensor_4D[0, :, :, :] = torch.FloatTensor(image_resize.transpose(2, 0, 1))
    inputs = tensor_4D.to(device)
    # -----------------------------------------------------------------
    _, alpha = net(inputs)
    return alpha

def get_out(image, alpha):
    origin_h, origin_w, c = image.shape
    if device == 'cpu':
        alpha_np = alpha[0, 0, :, :].data.numpy()
    else:
        alpha_np = alpha[0, 0, :, :].cpu().data.numpy()

    # cv2.resize的dsize参数是fx fy 对应W H
    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    # print('fg_alpha', fg_alpha) 0-1之间的小数

    # -----------------------------------------------------------------
    fg_alpha = fg_alpha[..., np.newaxis]
    bg = np.full(fg_alpha.shape, 255.0)
    # np.newaxis 添加一个维度做通道 比如(3,) -> (3,1)
    out = np.multiply(fg_alpha, image) + np.multiply(bg, (1 - fg_alpha))  # 得到合成图片

    # fg : color
    # out = fg
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8)
    return out

def ofd(pre_seg, cur_seg, next_seg):
    one = torch.ones_like(pre_seg)
    zero = torch.zeros_like(pre_seg)
    t1 = torch.abs(pre_seg - next_seg)
    t1 = torch.where(t1 <= EPSILON, one, zero)
    t2 = abs(cur_seg - pre_seg)
    t2 = torch.where(t2 > EPSILON, one, zero)
    t3 = abs(cur_seg - next_seg)
    t3 = torch.where(t3 > EPSILON, one, zero)
    t4 = t1.mul(t2).mul(t3) #三个条件都满足才能是1
    t4_sum = torch.sum(t4) / t4.numel()
    if t4_sum < 0.05:
        res = torch.where(t4 == 1, (pre_seg + next_seg) * 0.5, cur_seg)
    else:
        res = cur_seg
    # res = torch.where(t4 == 1, one, cur_seg)
    return res


def video_seg(video, result, net, fps=30):

    videoCapture = cv2.VideoCapture(video) #参数0表示打开内置摄像头

    num_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, pre_frame = videoCapture.read()

    pre_frame = cv2.flip(pre_frame, 1)
    pre_seg = get_alpha(pre_frame, net)

    ret, cur_frame = videoCapture.read()
    cur_frame = cv2.flip(cur_frame, 1)
    cur_seg = get_alpha(pre_frame, net)

    next_frame = None
    next_seg = None

    h, w = pre_frame.shape[:2]
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))

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

        # 存
        video_writer.write(res)
    videoCapture.release()
    print('Save the result video to {0}'.format(result))




if __name__ == "__main__":
    myModel = load_model(model_path)
    result = os.path.splitext(video_path)[0] + '_{0}.mp4'.format('ofd_result')
    video_seg(video_path, result, myModel)


