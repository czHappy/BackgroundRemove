import torch
import torch.nn as nn

class CNN(nn.Module):  # 继承了nn.Model
    def __init__(self):
        # 继承了 CNN 父类的的属性
        super(CNN, self).__init__()  # 用父类的初始化方式来初始化所继承的来自父类的属性
        # 按照网络的前后顺序定义1号网络
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(  # 这里的nn.Conv2d使用一个2维度卷积
                in_channels=1,  # in_channels：在文本应用中，即为词向量的维度
                out_channels=16,  # out_channels：卷积产生的通道数，有多少个out_channels，就需要多少个一维卷积（也就是卷积核的数量）
                kernel_size=5,
                # kernel_size：卷积核的尺寸；卷积核的第二个维度由in_channels决定，所以实际上卷积核的大小为kernel_size * in_channels
                stride=1,  # 步长，每次移动的单位格子
                padding=2,  # padding：对输入的每一条边，补充0的层数
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # 激活函数ReLU   # activation
            # 在2X2的池化层里选出最大值
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        # 按照网络的前后顺序定义2号网络，
        self.conv2 = nn.Sequential(  # input shape (1, 28, 28)
            # 使用一个二维卷积
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 因为在pytorch中做全连接的输入输出都是二维张量，不同于卷积层要求输入4维张量

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        # output是我们的真实值，而x是用于做数据可视化的参数
        return output, x  # return x for visualization