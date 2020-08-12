import torchvision
import torchvision.models as models
import torch
from torch import optim
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import DNN.transfer_learning.VGG as VGG


def modify_vgg(model, class_num):

    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, class_num))


BATCH_SIZE = 4
MODEL_PATH = './DNN/models/VGG16_bn.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_PATH = r'./dataset/cats_and_dogs/val'

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),#  The ToTensor transform should come before the Normalize transform, since the latter expects a tensor, but the Resize transform returns an image.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # 可以选做数据增强
    # transforms.RandomRotation(20),
    # transforms.RandomHorizontalFlip(),
])


test_dataset = torchvision.datasets.ImageFolder(root=TEST_PATH,
                                                 transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

def test(model, device, test_loader):
    model.eval()# eval 模式 关闭 BatchNormalization 和 Dropout
    correct = 0
    with torch.no_grad(): #关闭梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print('out = ', output.cpu().numpy().shape) # numpy只能在cpu上工作
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            # print('pred = ', pred.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                               100. * correct / len(test_loader.dataset)))

# 在vgg模块里找下载在本地的文件和model对应vgg的关系 有多仲vgg 不对应好就得下载
# 这样读取模型比较方便
# model = models.vgg16_bn(pretrained=False)

# 不用torchvision里的 自定义的VGG网络




#import DNN.ClassicalNet as ClassicalNet
#model_load = ClassicalNet.AlexNet().to(DEVICE) #测验网络不匹配 Missing key(s) in state_dict Unexpected key(s) in state_dict
#print('model : ', model_load)
model = VGG.vgg16_bn(pretrained=False, progress=False)
modify_vgg(model, class_num=2) #必须要将网络结构与保存的模型定义成一致的

model_load = model.to(DEVICE)

model_load.load_state_dict(torch.load(MODEL_PATH)) #load参数
print(model_load)
test(model_load, DEVICE, test_loader)


#model_load = ClassicalNet.AlexNet().to(DEVICE)
#model_load.load_state_dict(torch.load(MODEL_PATH))

# 断点续训