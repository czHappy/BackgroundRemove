'''
Author  : Zhengwei Li
Version : 1.0.0 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# kernel=3, stride = 1, padding = 1 结果的size不变
# 卷积 归一化 激活
def conv_bn_act(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        #nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False, padding_mode=None),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup)
    )

# 归一化 激活
def bn_act(inp):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        nn.PReLU(inp)
    )

# X concat CBA(X) 特征图size不变 通道数+growthRate
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(make_dense, self).__init__()
        # 默认的dilation为1，就是普通的卷积操作，如果dilation大于1，表示此时进行空洞卷积
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, dilation=1, bias=False) #stride默认1
        self.bn = nn.BatchNorm2d(growthRate)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x_ = self.bn(self.conv(x))
        out = self.act(x_)
        out = torch.cat((x, out), 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, reset_channel=False):
        super(DenseBlock, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate)) #堆叠dense层，通道数增加growthRate，特征图大小不变
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules) # Sequential传入多个模块 用可变参数

    def forward(self, x):
        out = self.dense_layers(x)
        return out

# ResidualDenseBlock 结果是size不变 channels不变
class ResidualDenseBlock(nn.Module):
    def __init__(self, nIn, s=4, add=True):

        super(ResidualDenseBlock, self).__init__()

        n = int(nIn//s)  # d = N/K paper中的K相当于这里的s

        self.conv = nn.Conv2d(nIn, n, 1, stride=1, padding=0, bias=False) #1x1卷积

        # s-1次dense 每次增加通道数n 通道数不变
        self.dense_block = DenseBlock(n, nDenselayer=(s-1), growthRate=n)

        self.bn = nn.BatchNorm2d(nIn)
        self.act = nn.PReLU(nIn)

        self.add = add

    def forward(self, input):

        # reduce
        inter = self.conv(input) #先降维到nIn//s 相当于减少通道为原来的1/s size不变
        combine = self.dense_block(inter) # 通道数不变 因为grow了s-1次 n+n(s-1) = ns = nIn size不变

        # if residual version A skip-connection between
        # input and output is added to improve the information flow
        if self.add:
            combine = input + combine #把输入和dense_block结果算术加 这相当于残差连接

        output = self.act(self.bn(combine))
        return output

class InputProjection(nn.Module):
    #卷积后，池化后尺寸计算公式：
    #(图像尺寸 - 卷积核尺寸 + 2 * 填充值) / 步长 + 1
    #(图像尺寸 - 池化窗尺寸 + 2 * 填充值) / 步长 + 1
    # 参阅https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1)) # 池化后 n/2上取整

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


# =========================================================================================
# 
# ESP  + Matting
# 
# =========================================================================================

class ERD_SegNet(nn.Module):

    def __init__(self, classes=2):

        super(ERD_SegNet, self).__init__()

        # -----------------------------------------------------------------
        # encoder 
        # ---------------------

        # input cascade
        self.cascade1 = InputProjection(1)
        self.cascade2 = InputProjection(2)
        self.cascade3 = InputProjection(3)
        self.cascade4 = InputProjection(4)

        # 1/2
        self.head_conv = conv_bn_act(3, 12, kernel_size=3, stride=2, padding=1) # ksp=321 ==> 2倍下采样
        self.stage_0 = ResidualDenseBlock(12, s=3, add=True) #

        # 1/4
        self.ba_1 = bn_act(12+3)
        self.down_1 = conv_bn_act(12+3, 24, kernel_size=3, stride=2, padding=1)
        self.stage_1 = ResidualDenseBlock(24, s=3, add=True)

        # 1/8
        self.ba_2 = bn_act(48+3)
        self.down_2 = conv_bn_act(48+3, 48, kernel_size=3, stride=2, padding=1)
        self.stage_2 = ResidualDenseBlock(48, s=3, add=True)

        # 1/16
        self.ba_3 = bn_act(96+3)
        self.down_3 = conv_bn_act(96+3, 96, kernel_size=3, stride=2, padding=1)
        self.stage_3 = nn.Sequential(ResidualDenseBlock(96, s=6, add=True),
                                     ResidualDenseBlock(96, s=6, add=True))
        # 1/32
        self.ba_4 = bn_act(192+3)
        self.down_4 = conv_bn_act(192+3, 192, kernel_size=3, stride=2, padding=1)
        self.stage_4 = nn.Sequential(ResidualDenseBlock(192, s=6, add=True),
                                     ResidualDenseBlock(192, s=6, add=True)) 

        # -----------------------------------------------------------------
        # heatmap 
        # ---------------------
        self.classifier = nn.Conv2d(192, classes, 1, stride=1, padding=0, bias=False)

        # -----------------------------------------------------------------
        # decoder 
        # ---------------------

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.prelu = nn.PReLU(classes) #Leaky ReLU 负区间斜率k可学习 k=0 => relu  k固定 => leaky relu

        self.stage3_down = conv_bn_act(96, classes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(classes)
        self.conv_3 = nn.Conv2d(classes, classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.stage2_down = conv_bn_act(48, classes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(classes)
        self.conv_2 = nn.Conv2d(classes, classes, kernel_size=3, stride=1, padding=1, bias=False)
 
        self.stage1_down = conv_bn_act(24, classes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(classes)
        self.conv_1 = nn.Conv2d(classes, classes, kernel_size=3, stride=1, padding=1, bias=False)  

        self.stage0_down = conv_bn_act(12, classes, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(classes)
        self.conv_0 = nn.Conv2d(classes, classes, kernel_size=3, stride=1, padding=1, bias=False)  
            
        self.last_up = nn.Upsample(scale_factor=2, mode='bilinear')


        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):

        input_cascade1 = self.cascade1(input)
        input_cascade2 = self.cascade2(input)
        input_cascade3 = self.cascade3(input)
        input_cascade4 = self.cascade4(input)

        x = self.head_conv(input) #3->12 1/2
        # 1/2
        s0 = self.stage_0(x)  #ResidualDenseBlock s=3 -> c=12 1/2

        # ---------------
        s1_0 = self.down_1(self.ba_1(torch.cat((input_cascade1, s0),1))) #1/2特征图concat ，in=12+3 out = 24 1/4
        s1 = self.stage_1(s1_0) # c=24 1/4

        # ---------------
        s2_0 = self.down_2(self.ba_2(torch.cat((input_cascade2, s1_0, s1),1))) # 3+24+24 out=48  1/8
        s2 = self.stage_2(s2_0) # out=48  1/8

        # ---------------
        s3_0 = self.down_3(self.ba_3(torch.cat((input_cascade3, s2_0, s2),1))) #1/8特征图concat in=3+48+48 out=96 1/16
        s3 = self.stage_3(s3_0) # out=96 1/16

        # ---------------
        s4_0 = self.down_4(self.ba_4(torch.cat((input_cascade4, s3_0, s3),1))) #1/16特征图concat in=3+96+96 out=192 1/32
        s4 = self.stage_4(s4_0) # out=192 1/32


        # -------------------------------------------------------

        heatmap = self.classifier(s4) # out=classes 1/32
        # -------------------------------------------------------

        # stagex_down 把通道数变为classes size不变

        heatmap_3 = self.up(heatmap) # 2x  out=classes 1/16
        s3_heatmap = self.prelu(self.bn3(self.stage3_down(s3))) # s3:c=96 1/16  out:c=classes 1/16
        heatmap_3 = heatmap_3 + s3_heatmap # out:c=classes 1/16
        heatmap_3 = self.conv_3(heatmap_3)# out:c=classes 1/16

        heatmap_2 = self.up(heatmap_3) # out=classes 1/8
        s2_heatmap = self.prelu(self.bn2(self.stage2_down(s2))) # s2:c=48 1/8  out:c=classes 1/8
        heatmap_2 = heatmap_2 + s2_heatmap # out=classes 1/8
        heatmap_2 = self.conv_2(heatmap_2) # out:c=classes 1/8

        heatmap_1 = self.up(heatmap_2) # out=classes 1/4
        s1_heatmap = self.prelu(self.bn1(self.stage1_down(s1))) # s1:c=24 1/4 out: c=classes 1/4
        heatmap_1 = heatmap_1 + s1_heatmap #out: c=classes 1/4
        heatmap_1 = self.conv_1(heatmap_1)   #out: c=classes 1/4

        heatmap_0 = self.up(heatmap_1) # out: c=classes 1/2
        s0_heatmap = self.prelu(self.bn0(self.stage0_down(s0))) #s0:c=12 1/2  out: c=classes 1/2
        heatmap_0 = heatmap_0 + s0_heatmap #out: c=classes 1/2
        heatmap_0 = self.conv_0(heatmap_0)   #out: c=classes 1/2

        out = self.last_up(heatmap_0) #out: c=classes original size

        return out

###################################################################################################
'''

      Segnet + Matting

feature extracter:
                    ERD_SegNet
                    ... ...

Matting:            filter  block

'''

class SegMattingNet(nn.Module):
    def __init__(self):
        super(SegMattingNet, self).__init__()


        self.seg_extract = ERD_SegNet(classes=2)

        # feather
        self.convF1 = nn.Conv2d(in_channels=11, out_channels=8, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn = nn.BatchNorm2d(num_features=8)
        self.ReLU = nn.ReLU(inplace=True)
        self.convF2 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        

        # init weights
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        
        seg = self.seg_extract(x) #seg是有classes个通道 这里classes = 2
        # shape: n 1 h w
        seg_softmax = F.softmax(seg, dim=1)
        bg, fg = torch.split(seg_softmax, 1, dim=1)

        # shape: n 3 h w
        imgSqr = x * x # I*I paper : Fast Deep Matting for Portrait....
        imgMasked = x * (torch.cat((fg, fg, fg), 1))
        # shape: n 11 h w
        convIn = torch.cat((x, seg_softmax, imgSqr, imgMasked), 1) # 3+2+3+3
        newconvF1 = self.ReLU(self.bn(self.convF1(convIn)))
        newconvF2 = self.convF2(newconvF1)
        
        # fethering inputs:
        a, b, c = torch.split(newconvF2, 1, dim=1)

        #print("seg: {}".format(seg))
        alpha = a * fg + b * bg + c        
        alpha = self.sigmoid(alpha)

        return seg, alpha
