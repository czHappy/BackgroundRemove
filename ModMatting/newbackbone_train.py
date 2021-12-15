import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from functools import reduce

import cv2
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.ndimage import morphology
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import copy
import logging
import shutil

logging.basicConfig(filename='Default.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('--------------------------------')

# ----------------------------------------------------------------------------------
# Tool Classes/Functions
# ----------------------------------------------------------------------------------
torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# make `v` is divided exactly by `divisor`, but keep the min_value
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# conv + bn
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), # conv 3X3 + padding = 1
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 1x1 conv
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# 而Instance Normalization是指单张图片的单个通道单独进行Noramlization操作 IN适用于生成模型中，比如图片风格迁移。因为图片生成的结果主要依赖于某个图像实例 沿着通道计算每个图均值和方差。
class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True) #affine=True进行仿射变换 有伽马贝塔参数

        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x): #一半BN 一半IN
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous()) # contiguous操作保证tensor内存行优先进行连续排列。
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1) #按通道concat起来


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation,
                      groups=groups, bias=bias)
        ]

        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf

    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)# 相比 nn.AvgPool2d() 多了个自适应，自适应就代表了使用更简单方便 参数是output-size
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        #print('x_size', x.size())
        w = self.pool(x).view(b, c)
        #print('w_size', w.size())
        #print('fcw = ', self.fc(w).size())
        w = self.fc(w).view(b, c, 1, 1)
        #print('wfc_size', w.size())

        return x * w.expand_as(x)


class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone):
        super(LRBranch, self).__init__()

        enc_channels = backbone.enc_channels
        self.backbone = backbone
        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False,
                                        with_relu=False)

    def forward(self, img, inference=True):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)

        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x) # S(I)
        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x) #S(I) feed to a conv layer to reduce channels to 1.
            pred_semantic = torch.sigmoid(lr) # Sp
        return pred_semantic, lr8x, [enc2x, enc4x]


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(HRBranch, self).__init__()

        self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.tohr_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 1, stride=1, padding=0)#add
        self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)
        
        self.conv_hr4x = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference=True):
        #print('img', img.size())

        img2x = F.interpolate(img, scale_factor=1 / 2, mode='bilinear', align_corners=False)
        #print('img2x', img2x.size())
        img4x = F.interpolate(img, scale_factor=1 / 4, mode='bilinear', align_corners=False)
        #print('img4x', img4x.size())
        enc2x = self.tohr_enc2x(enc2x)
        #print('enc2x', enc2x.size())
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))
        #print('hr4x', hr4x.size())
        enc4x = self.tohr_enc4x(enc4x)
        #print('enc4x', enc4x.size())
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        #print('hr4x', hr4x.size())
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.tohr_lr4x(lr4x) #这里做一下兼容 将lr4x通道数改成hr通道数
        #print('lr4x', lr4x.size())
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))
        #print('hr4x', hr4x.size())

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        #print('hr2x', hr2x.size())
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))
        #print('hr2x', hr2x.size())

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr) # dp
            #print('dp', pred_detail.size())
        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(FusionBranch, self).__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)

        self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)
        return pred_matte

from resnet101 import *
class MODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=False):
        super(MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        #self.backbone = ResNet101Backbone(self.in_channels)
        self.backbone = ResNet34Backbone(self.in_channels)
        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()

    def forward(self, img, inference=True):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)

        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte

    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)


class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input
        """
        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))


class ImagesDataset(Dataset):
    """because of i want to use the modle author provided, i use the same size as the val in demo,
    if you want to change the size, change the size in code func: __getitem__, attention the shape"""
    def __init__(self, root, transform=None, w=1024, h=576):
        self.root = root
        self.transform = transform
        self.tensor = transforms.Compose([transforms.ToTensor()])
        self.w = w
        self.h = h
        self.imgs = sorted(os.listdir(os.path.join(self.root, 'imgs')))
        self.alphas = sorted(os.listdir(os.path.join(self.root, 'alphas')))
        assert len(self.imgs) == len(self.alphas), 'the number of dataset is different, please check it.'

    def getTrimap(self, alpha):
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20)
        trimap = fg
        trimap[unknown] = 0.5
        # print(trimap[:, :, :1].shape)
        return trimap[:, :, :1]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root, 'imgs', self.imgs[idx]))
        alpha = cv2.imread(os.path.join(self.root, 'alphas', self.alphas[idx]))
        h, w, c = img.shape
        rh = 512
        #rw = int(w / h * 512)
        #rh = rh - rh % 32  # 512
        #rw = rw - rw % 32  # 896
        rw = 896
        img = cv2.resize(img, (rw, rh))
        alpha = cv2.resize(alpha, (rw, rh))
        trimap = self.getTrimap(alpha)
        # print(trimap.shape)
        if self.transform:
            img = self.transform(img)
        alpha = self.tensor(alpha[:, :, 0])
        return self.imgs[idx], img, trimap, alpha


# ----------------------------------------------------------------------------------
class ImagesDatasetSOC(Dataset):
    """make it like class ImagesDataset"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.tensor = transforms.Compose([transforms.ToTensor()])
        self.imgs = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.root, self.imgs[idx]))
        h, w, c = img.shape
        rh = 512
        rw = int(w / h * 512)
        rh = rh - rh % 32  #
        rw = rw - rw % 32  #
        img = cv2.resize(img, (rw, rh))

        if self.transform:
            img = self.transform(img)

        return self.imgs[idx], img
# ----------------------------------------------------------------------------------
# MODNet Training Functions
# ----------------------------------------------------------------------------------

#blurer = GaussianBlurLayer(1, 3).cuda()
blurer = GaussianBlurLayer(1, 3)

def supervised_training_iter(
        modnet, optimizer, image, trimap, gt_matte,
        semantic_scale=10.0, detail_scale=10.0, matte_scale=1.0):
    """ Supervised training iteration of MODNet
    This function trains MODNet for one iteration in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        optimizer (torch.optim.Optimizer): optimizer for supervised training
        image (torch.autograd.Variable): input RGB image
        trimap (torch.autograd.Variable): trimap used to calculate the losses
                                          NOTE: foreground=1, background=0, unknown=0.5
        gt_matte (torch.autograd.Variable): ground truth alpha matte
        semantic_scale (float): scale of the semantic loss
                                NOTE: please adjust according to your dataset
        detail_scale (float): scale of the detail loss
                              NOTE: please adjust according to your dataset
        matte_scale (float): scale of the matte loss
                             NOTE: please adjust according to your dataset

    Returns:
        semantic_loss (torch.Tensor): loss of the semantic estimation [Low-Resolution (LR) Branch]
        detail_loss (torch.Tensor): loss of the detail prediction [High-Resolution (HR) Branch]
        matte_loss (torch.Tensor): loss of the semantic-detail fusion [Fusion Branch]

    Example:
        import torch
        from src.models.modnet import MODNet
        from src.trainer import supervised_training_iter

        bs = 16         # batch size
        lr = 0.01       # learn rate
        epochs = 40     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)

        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            for idx, (image, trimap, gt_matte) in enumerate(dataloader):
                semantic_loss, detail_loss, matte_loss = \
                    supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            lr_scheduler.step()
    """

    global blurer
    # set the model to train mode and clear the optimizer
    modnet.train()
    optimizer.zero_grad()

    # forward the model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # calculate the boundary mask from the trimap
    boundaries = (trimap < 0.5) + (trimap > 0.5)

    # calculate the semantic loss
    gt_semantic = F.interpolate(gt_matte, scale_factor=1 / 16, mode='bilinear')
    gt_semantic = blurer(gt_semantic)
    semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
    semantic_loss = semantic_scale * semantic_loss

    # calculate the detail loss
    pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
    gt_detail = torch.where(boundaries, trimap, gt_matte)
    detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
    matte_l1_loss = F.l1_loss(pred_matte, gt_matte) + 4.0 * F.l1_loss(pred_boundary_matte, gt_matte)
    matte_compositional_loss = F.l1_loss(image * pred_matte, image * gt_matte) \
                               + 4.0 * F.l1_loss(image * pred_boundary_matte, image * gt_matte)
    matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
    matte_loss = matte_scale * matte_loss

    # calculate the final loss, backward the loss, and update the model
    loss = semantic_loss + detail_loss + matte_loss
    loss.backward()
    optimizer.step()

    # for test
    return semantic_loss, detail_loss, matte_loss


def soc_adaptation_iter(
        modnet, backup_modnet, optimizer, image,
        soc_semantic_scale=100.0, soc_detail_scale=1.0):
    """ Self-Supervised sub-objective consistency (SOC) adaptation iteration of MODNet
    This function fine-tunes MODNet for one iteration in an unlabeled dataset.
    Note that SOC can only fine-tune a converged MODNet, i.e., MODNet that has been
    trained in a labeled dataset.

    Arguments:
        modnet (torch.nn.Module): instance of MODNet
        backup_modnet (torch.nn.Module): backup of the trained MODNet
        optimizer (torch.optim.Optimizer): optimizer for self-supervised SOC
        image (torch.autograd.Variable): input RGB image
        soc_semantic_scale (float): scale of the SOC semantic loss
                                    NOTE: please adjust according to your dataset
        soc_detail_scale (float): scale of the SOC detail loss
                                  NOTE: please adjust according to your dataset

    Returns:
        soc_semantic_loss (torch.Tensor): loss of the semantic SOC
        soc_detail_loss (torch.Tensor): loss of the detail SOC

    Example:
        import copy
        import torch
        from src.models.modnet import MODNet
        from src.trainer import soc_adaptation_iter

        bs = 1          # batch size
        lr = 0.00001    # learn rate
        epochs = 10     # total epochs

        modnet = torch.nn.DataParallel(MODNet()).cuda()
        modnet = LOAD_TRAINED_CKPT()    # NOTE: please finish this function

        optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
        dataloader = CREATE_YOUR_DATALOADER(bs)     # NOTE: please finish this function

        for epoch in range(0, epochs):
            backup_modnet = copy.deepcopy(modnet)
            for idx, (image) in enumerate(dataloader):
                soc_semantic_loss, soc_detail_loss = \
                    soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
    """

    global blurer

    # set the backup model to eval mode
    backup_modnet.eval()

    # set the main model to train mode and freeze its norm layers
    modnet.train()
    modnet.module.freeze_norm()

    # clear the optimizer
    optimizer.zero_grad()

    # forward the main model
    pred_semantic, pred_detail, pred_matte = modnet(image, False)

    # forward the backup model
    with torch.no_grad():
        _, pred_backup_detail, pred_backup_matte = backup_modnet(image, False)

    # calculate the boundary mask from `pred_matte` and `pred_semantic`
    pred_matte_fg = (pred_matte.detach() > 0.1).float()
    pred_semantic_fg = (pred_semantic.detach() > 0.1).float()
    pred_semantic_fg = F.interpolate(pred_semantic_fg, scale_factor=16, mode='bilinear')
    pred_fg = pred_matte_fg * pred_semantic_fg

    n, c, h, w = pred_matte.shape
    np_pred_fg = pred_fg.data.cpu().numpy()
    np_boundaries = np.zeros([n, c, h, w])
    for sdx in range(0, n):
        sample_np_boundaries = np_boundaries[sdx, 0, ...]
        sample_np_pred_fg = np_pred_fg[sdx, 0, ...]

        side = int((h + w) / 2 * 0.05)
        dilated = grey_dilation(sample_np_pred_fg, size=(side, side))
        eroded = grey_erosion(sample_np_pred_fg, size=(side, side))

        sample_np_boundaries[np.where(dilated - eroded != 0)] = 1
        np_boundaries[sdx, 0, ...] = sample_np_boundaries

    boundaries = torch.tensor(np_boundaries).float().cuda()

    # sub-objectives consistency between `pred_semantic` and `pred_matte`
    # generate pseudo ground truth for `pred_semantic`
    downsampled_pred_matte = blurer(F.interpolate(pred_matte, scale_factor=1 / 16, mode='bilinear'))
    pseudo_gt_semantic = downsampled_pred_matte.detach()
    pseudo_gt_semantic = pseudo_gt_semantic * (pseudo_gt_semantic > 0.01).float()

    # generate pseudo ground truth for `pred_matte`
    pseudo_gt_matte = pred_semantic.detach()
    pseudo_gt_matte = pseudo_gt_matte * (pseudo_gt_matte > 0.01).float()

    # calculate the SOC semantic loss
    soc_semantic_loss = F.mse_loss(pred_semantic, pseudo_gt_semantic) + F.mse_loss(downsampled_pred_matte,
                                                                                   pseudo_gt_matte)
    soc_semantic_loss = soc_semantic_scale * torch.mean(soc_semantic_loss)

    # NOTE: using the formulas in our paper to calculate the following losses has similar results
    # sub-objectives consistency between `pred_detail` and `pred_backup_detail` (on boundaries only)
    backup_detail_loss = boundaries * F.l1_loss(pred_detail, pred_backup_detail)
    backup_detail_loss = torch.sum(backup_detail_loss, dim=(1, 2, 3)) / torch.sum(boundaries, dim=(1, 2, 3))
    backup_detail_loss = torch.mean(backup_detail_loss)

    # sub-objectives consistency between pred_matte` and `pred_backup_matte` (on boundaries only)
    backup_matte_loss = boundaries * F.l1_loss(pred_matte, pred_backup_matte)
    backup_matte_loss = torch.sum(backup_matte_loss, dim=(1, 2, 3)) / torch.sum(boundaries, dim=(1, 2, 3))
    backup_matte_loss = torch.mean(backup_matte_loss)

    soc_detail_loss = soc_detail_scale * (backup_detail_loss + backup_matte_loss)

    # calculate the final loss, backward the loss, and update the model
    loss = soc_semantic_loss + soc_detail_loss

    loss.backward()
    optimizer.step()

    return soc_semantic_loss, soc_detail_loss


# ----------------------------------------------------------------------------------


def main(root, resume=False, std=1):
    """ resume=True if not first runing else False  """
    save_model_dir = './101'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    print(torch.cuda.is_available())
    if resume:
        VModel = sorted(os.listdir(save_model_dir))[-1]
        pretrained_ckpt = os.path.join(save_model_dir, VModel)
    else:
        pretrained_ckpt = r'./modnet_photographic_portrait_matting.ckpt'
    #print(pretrained_ckpt)
    #logging.info(f"model load {pretrained_ckpt}")
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        # Comment next row out if you change the image size or you don't use the modle author provided else pass
        #modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        # Comment next row out if you change the image size or you don't use the modle author provided else pass
        modnet.load_state_dict(torch.load(pretrained_ckpt, map_location=torch.device('cpu')))
    bs = 16  # batch size
    lr = 0.001  # learn rate
    epochs = 401  # total epochs
    num_workers = 16
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60,
                                                   gamma=0.1)  # step_size 学习率下降迭代间隔次数， default: 每10次降低一次学习率

    # dataloader = CREATE_YOUR_DATALOADER(bs)  # NOTE: please finish this function
    dataset = ImagesDataset(root, torch_transforms)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True, shuffle=True)

    for epoch in range(std, epochs+1):
        mattes = []
        for idx, (img_file, image, trimap, gt_matte) in enumerate(dataloader, start=1):
            trimap = np.transpose(trimap, (0, 3, 1, 2)).float().cuda()
            image = image.cuda()
            gt_matte = gt_matte.cuda()
            # print(image.shape)
            # print(trimap.shape)
            # print(gt_matte.shape)
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap,
                                                                              gt_matte)
            info = f"epoch: {epoch}/{epochs} semantic_loss: {semantic_loss}, detail_loss: {detail_loss}, matte_loss： {matte_loss}"
            print(idx, info)
            logging.info(info)
            mattes.append(float(matte_loss))
        avg_matte = float(np.mean(mattes))
        logging.info(f"epoch: {epoch}/{epochs}, average_matte_loss: {avg_matte}")
        lr_scheduler.step()
        if epoch % 20 == 0:
            torch.save(modnet.state_dict(), os.path.join(save_model_dir, '{}.ckpt'.format(epoch)))
            print(f'------save model------{epoch}  {epoch}.ckpt')
            logging.info(f'------save model------{epoch}  {epoch}.ckpt')


def mainSoc(root):
    save_model_dir = "./"
    """make it like main()"""
    bs = 2  # batch size
    lr = 0.00001  # learn rate
    epochs = 10  # total epochs
    modnet = torch.nn.DataParallel(MODNet()).cuda()
    pretrained_ckpt = './modnet_webcam_portrait_matting.ckpt'
    modnet.load_state_dict(torch.load(pretrained_ckpt))
    print("load model weight complete!")
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                   gamma=0.1)  # step_size 学习率下降迭代间隔次数， default: 每10次降低一次学习率
    num_workers = 2
    dataset = ImagesDatasetSOC(root, torch_transforms)
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=num_workers, pin_memory=True)
    print(len(dataset))
    print("start train")
    for epoch in range(0, epochs):
        backup_modnet = copy.deepcopy(modnet)
        for idx, (img_file, image) in enumerate(dataloader):
            print(img_file)
            image = image.cuda()
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
            print('epoch:', epoch, ' batch:', idx, ' soc_semantic_loss = ', soc_semantic_loss)
            print('epoch:', epoch, ' batch:', idx, ' soc_detail_loss = ', soc_detail_loss)

        lr_scheduler.step()

    if epoch % 2 == 0:
        torch.save(modnet.state_dict(), os.path.join(save_model_dir, 'soc_{:0>6d}.ckpt'.format(epoch)))
        print(f'------save model------{epoch}  {epoch}.ckpt')
        logging.info(f'------save model------{epoch}  {epoch}.ckpt')


if __name__ == '__main__':
    path1 = r'/home/chengzhen/mobile-code-bak/dataset'
    # step1
    main(path1)
    # step2, run it after finish step1
    #path2 = r"E:\Data_Backup\mobile-torch1.x\dataset\soc_imgs"
    #mainSoc(path2)
    # mainSoc()
