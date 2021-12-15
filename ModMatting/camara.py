import cv2
import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from main import MODNet
from calc_summary import count_params, calc_flops




torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
#pretrained_ckpt = './origin/modnet_webcam_portrait_matting.ckpt'
pretrained_ckpt = './origin/modnet_photographic_portrait_matting.ckpt'
#pretrained_ckpt = './90.ckpt'
modnet = MODNet(backbone_pretrained=True)


GPU = True if torch.cuda.device_count() > 0 else False
# GPU = False
if GPU:
    print('Use GPU...')
    modnet = nn.DataParallel(modnet)
    #modnet = modnet.cuda()
    modnet.load_state_dict(torch.load(pretrained_ckpt))
    modnet.eval()
    modnet.to('cuda')

    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(modnet, (3, 512, 512), as_strings=True, print_per_layer_stat=True)
    #
    # print("%s |flops: %s |params: %s" % ('modnet', flops, params))



else:
    print('Use CPU...')
    # modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    # modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=lambda storage, loc: storage).module)
    state_dict = torch.load(pretrained_ckpt, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    modnet.load_state_dict(new_state_dict)
    modnet.to('cpu')

    # from torchstat import stat
    # stat(modnet, (3, 512, 512))
modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture(0) #创建VideoCapture，传入0即打开系统默认摄像头
W = 640
H = 480
W = W - W % 32
H = H - H % 32
print(W, H)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5*W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5*H)

#print(count_params(modnet, HEIGHT=H, WIDTH=W))

print('Start matting...')
while (True):
    start = time.time()
    _, frame_np = cap.read()
    frame_np = cv2.flip(frame_np, 1)
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    frame_np = cv2.resize(frame_np, (W, H), cv2.INTER_AREA)
    # frame_np = frame_np[16:, :, :]
    print(frame_np.shape)
    # frame_np = cv2.flip(frame_np, 1) #水平翻转

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    #print(frame_tensor.shape)
    frame_tensor = frame_tensor[None, :, :, :] # None就是加一个维度变成标准4Dtensor
    #print(frame_tensor.shape)
    if GPU:
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        st = time.time()
        _, _, matte_tensor = modnet(frame_tensor, True)
        ed = time.time()
        print("pure fps = ", 1.0 / (ed - st))
    print(matte_tensor.shape)
    matte_tensor = matte_tensor.repeat(1, 3, 1, 1) #NCHW
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0) #hwc
    fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
    # view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1)) #在w维度concatenate
    fg_np = cv2.resize(fg_np, (640, 480), cv2.INTER_AREA)
    view_np = np.uint8(fg_np)
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    end = time.time()
    print("fps = ", 1.0/(end - start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Exit...')


