import torch.nn as nn
import torch.utils.model_zoo as model_zoo



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

}

class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = [self.conv1, self.bn1, self.relu]
        self.layer0 = nn.Sequential(*self.layer0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 到这里输出应该是7X7X2048  原来是7X7X1280


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0[x]
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _load_pretrained_model(self, pretrain_dict):
        # pretrain_dict = torch.load(pretrained_file, map_location='cpu')
        # pretrain_dict = torch.load(pretrained_file)
        model_dict = {}
        state_dict = self.state_dict()
        print("[Resnet] Loading pretrained model...")
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
            else:
                print(k, " is ignored")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNet101Backbone(BaseBackbone):
    def __init__(self, in_channels):
        super(ResNet101Backbone, self).__init__(in_channels)
        self.model = ResNet(Bottleneck, [3, 4, 23, 3])
        self.enc_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        x = self.model.layer0(x)
        enc2x = x
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x
        x = self.model.layer2(x)
        enc8x = x
        x = self.model.layer3(x)
        enc16x = x
        x = self.model.layer4(x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        self.model._load_pretrained_model(model_zoo.load_url(model_urls['resnet101']))


class ResNet50Backbone(BaseBackbone):
    def __init__(self, in_channels):
        super(ResNet50Backbone, self).__init__(in_channels)
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        self.enc_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        x = self.model.layer0(x)
        enc2x = x
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x
        x = self.model.layer2(x)
        enc8x = x
        x = self.model.layer3(x)
        enc16x = x
        x = self.model.layer4(x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        self.model._load_pretrained_model(model_zoo.load_url(model_urls['resnet50']))


class ResNet34Backbone(BaseBackbone):
    def __init__(self, in_channels):
        super(ResNet34Backbone, self).__init__(in_channels)
        self.model = ResNet(BasicBlock, [3, 4, 6, 3])
        self.enc_channels = [64, 64, 128, 256, 512]

    def forward(self, x):
        x = self.model.layer0(x)
        enc2x = x
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        enc4x = x
        x = self.model.layer2(x)
        enc8x = x
        x = self.model.layer3(x)
        enc16x = x
        x = self.model.layer4(x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        self.model._load_pretrained_model(model_zoo.load_url(model_urls['resnet34']))



