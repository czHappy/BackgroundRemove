'''
Author  : Zhengwei Li
Version : 1.0.0 
'''
import time
import cv2
import torch 
import pdb
import argparse
import numpy as np
import os 
from collections import OrderedDict
import torch.nn.functional as F
from model import segnet
parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./model/*.pt', help='preTrained model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--bg', default='./bg/bg.jpg', help='choose background image.')
args = parser.parse_args()

torch.set_grad_enabled(False) # 不要计算导数了在interfere的时候

INPUT_SIZE = 512
EPSILON = 0.2
# --model ./pre_trained/erd/model/model_obj.pth
# --model ./pre_trained/super/model_super_4/model_obj.pth

print(torch.__version__)

#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')



#################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/5
        # myModel = torch.load(args.model, map_location=lambda storage, loc: storage) #把GPU上训练的模型加载到CPU上
        #myModel = segnet.SegMattingNet()
        #myModel.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage)['state_dict'])
        #myModel = segnet.SegMattingNet()
        #myModel.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # myModel = segnet.SegMattingNet()
        # myModel.load_state_dict(torch.load(args.model)['state_dict'])
        # myModel = torch.load(args.model)
        myModel = segnet.SegMattingNet()
        myModel.load_state_dict(torch.load(args.model))
    myModel.eval() # 设置模型为eval模式
    myModel.to(device)

    return myModel
#


def get_alpha(image, net):
    # opencv

    # print('img shape = ', image.shape)
    image_resize = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0  # 归一化

    tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)  # 变单张图片为tensor

    # torch.FloatTensor 是torch.Tensor的简称
    tensor_4D[0, :, :, :] = torch.FloatTensor(image_resize.transpose(2, 0, 1))
    inputs = tensor_4D.to(device)
    # -----------------------------------------------------------------


    _, alpha = net(inputs)
    return alpha

def get_out(image, alpha):
    origin_h, origin_w, c = image.shape
    if args.without_gpu:
        alpha_np = alpha[0, 0, :, :].data.numpy()
    else:
        alpha_np = alpha[0, 0, :, :].cpu().data.numpy()

    # cv2.resize的dsize参数是fx fy 对应W H
    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    # print('fg_alpha', fg_alpha) 0-1之间的小数

    # -----------------------------------------------------------------
    # np.newaxis 添加一个维度做通道 比如(3,) -> (3,1)
    fg = np.multiply(fg_alpha[..., np.newaxis], image)  # 得到前景

    # gray
    bg = image
    bg_alpha = 1 - fg_alpha[..., np.newaxis]
    bg_alpha[bg_alpha < 0] = 0

    bg_gray = np.multiply(bg_alpha, image)  # 得到背景图
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)  # 得到背景灰度图

    bg[:, :, 0] = bg_gray
    bg[:, :, 1] = bg_gray
    bg[:, :, 2] = bg_gray

    # 替换任意背景
    bg_img = cv2.imread(args.bg)
    bg_img = cv2.resize(bg_img, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    bg_replace = np.multiply(1 - fg_alpha[..., np.newaxis], bg_img)
    bg = bg_replace

    # -----------------------------------------------------------------
    # fg : color, bg : gray
    out = fg + bg

    # fg : color
    # out = fg
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8)

    return out

def seg_process(args, image, net):

    # opencv
    origin_h, origin_w, c = image.shape
    # print('img shape = ', image.shape)
    image_resize = cv2.resize(image, (INPUT_SIZE,INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0 #归一化


    tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE) #变单张图片为tensor

    # torch.FloatTensor 是torch.Tensor的简称
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2, 0, 1))
    inputs = tensor_4D.to(device)
    # -----------------------------------------------------------------

    t0 = time.time()

    seg, alpha = net(inputs)


    #print('seg', seg)
    print((time.time() - t0))  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()

    # cv2.resize的dsize参数是fx fy 对应W H
    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    # print('fg_alpha', fg_alpha) 0-1之间的小数

    # -----------------------------------------------------------------
    # np.newaxis 添加一个维度做通道 比如(3,) -> (3,1)
    fg = np.multiply(fg_alpha[..., np.newaxis], image) #得到前景


    # gray
    bg = image
    bg_alpha = 1 - fg_alpha[..., np.newaxis]
    bg_alpha[bg_alpha<0] = 0

    bg_gray = np.multiply(bg_alpha, image) # 得到背景图
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)# 得到背景灰度图

    bg[:,:,0] = bg_gray
    bg[:,:,1] = bg_gray
    bg[:,:,2] = bg_gray

    # 替换任意背景
    bg_img = cv2.imread(args.bg)
    bg_img = cv2.resize(bg_img, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
    bg_replace = np.multiply(1 - fg_alpha[..., np.newaxis], bg_img)
    bg = bg_replace

    # -----------------------------------------------------------------
    # fg : color, bg : gray
    out = fg + bg

    # fg : color
    #out = fg
    out[out<0] = 0
    out[out>255] = 255
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
    res = torch.where(t4 == 1, (pre_seg + next_seg) * 0.5, cur_seg)
    # res = torch.where(t4 == 1, one, cur_seg)
    return res


def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0) #参数0表示打开内置摄像头
    ret, pre_frame = videoCapture.read()
    pre_frame = cv2.flip(pre_frame, 1)
    pre_seg = get_alpha(pre_frame, net)

    ret, cur_frame = videoCapture.read()
    cur_frame = cv2.flip(cur_frame, 1)
    cur_seg = get_alpha(pre_frame, net)

    next_frame = None
    next_seg = None
    while(1):
        # get a frame
        # ret, cur_frame = videoCapture.read()
        # cur_frame = cv2.flip(cur_frame, 1) # 0 means flipping around the x-axis and positive value (for example, 1) means flipping around y-axis.
        # cur_seg = get_alpha(cur_frame, net)
        ret, next_frame = videoCapture.read()
        next_frame = cv2.flip(next_frame, 1)
        next_seg = get_alpha(next_frame, net)
        res_seg = ofd(pre_seg, cur_seg, next_seg)
        # res_seg = cur_seg
        res = get_out(cur_frame, res_seg)

        pre_seg = cur_seg
        cur_seg = next_seg
        cur_frame = next_frame

        # show a frame
        cv2.imshow("capture", res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()


def main(args):
    myModel = load_model(args)
    camera_seg(args, myModel)

if __name__ == "__main__":
    main(args)


