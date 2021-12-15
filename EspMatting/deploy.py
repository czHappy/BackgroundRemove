'''
Author  : Zhengwei Li
Version : 1.0.0 
'''
from model import segnet
import time
import cv2
import torch 
import pdb
import argparse
import os 
import numpy as np
import torch.nn.functional as F
import pdb
import sys
from torchstat import stat
parser = argparse.ArgumentParser(description='Semantic aware super-resolution')
parser.add_argument('--model', default='./model/*.pt', help='preTrained model')
parser.add_argument('--inputPath', default='./', help='input data path')
parser.add_argument('--savePath', default='./', help='output data path')
parser.add_argument('--size', type=int, default=128, help='net input size')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')


args = parser.parse_args()

# --model ./pre_trained/erd/model/model_obj.pth
# --model ./pre_trained/super/model_super_4/model_obj.pth
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

# 获取网络残数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters()) # numel()返回数组元素数量
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/5
        # myModel = torch.load(args.model, map_location=lambda storage, loc: storage) #把GPU上训练的模型加载到CPU上
        myModel = segnet.SegMattingNet()
        myModel.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage)['state_dict'])
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # myModel = torch.load(args.model)
        myModel = segnet.SegMattingNet()
        myModel.load_state_dict(torch.load(args.model)['state_dict'])
        # myModel.load_state_dict(torch.load(args.model))
    myModel.eval()  # 设置模型为eval模式
    myModel.to(device)

    return myModel

def np_norm(x):
    low = x.min()
    hig = x.max()
    y = (x - low) / (hig - low)
    return y

def seg_process(args, net):
    filelist = [f for f in os.listdir(args.inputPath)]
    filelist.sort()
    # set grad false
    torch.set_grad_enabled(False)
    i = 0
    t_all = 0
    for f in filelist:
        print('The %dth image : %s ...'%(i,f))
        print(os.path.join(args.inputPath, f))
        image = cv2.imread(os.path.join(args.inputPath, f))
        #print(image)
        # image = image[:,400:,:]
        origin_h, origin_w, c = image.shape
        image_resize = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
        image_resize = (image_resize - (104., 112., 121.,)) / 255.0

        tensor_4D = torch.FloatTensor(1, 3, args.size, args.size)
        tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1)) #nchw
        inputs = tensor_4D.to(device)

        t0 = time.time()
        seg, alpha = net(inputs)

        if args.without_gpu:
            alpha_np = alpha[0,0,:,:].data.numpy()
        else:
            alpha_np = alpha[0,0,:,:].cpu().data.numpy()

        tt = (time.time() - t0)

        alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("./mask_test.png", (alpha_np*255).astype(np.uint8))
        seg_fg = np.multiply(alpha_np[..., np.newaxis], image)

        f = f[:-4] + '_result.png'
        cv2.imwrite(os.path.join(args.savePath, f), seg_fg)

        i+=1
        t_all += tt

    print("image number: {} mean matting time : {:.0f} ms".format(i, t_all/i*1000))

def main(args):

    myModel = load_model(args)
    seg_process(args, myModel)

if __name__ == "__main__":
    main(args)
