import torch
from DNN import ClassicalNet
import torchvision
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
BATCH_SIZE = 32
MODEL_PATH = r'./models/AlexNet.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_PATH = r'../../dataset/cats_and_dogs/val'
transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
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
            print('out = ', output.cpu().numpy()) # numpy只能在cpu上工作
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            print('pred = ', pred.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


#注意这里必须送到DEVICE中去 原因是cpu和gpu上的tensor不一样
#否则报错Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
model_load = ClassicalNet.AlexNet().to(DEVICE)
model_load.load_state_dict(torch.load(MODEL_PATH))
# model.train() ：启用 BatchNormalization 和 Dropout
# model.eval() ：不启用 BatchNormalization 和 Dropout
test(model_load, DEVICE, test_loader)

