
# models.vgg16 ==> Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth"
#  to C:\Users\cz/.cache\torch\checkpoints\vgg16-397923af.pth
# 说明下载的是vgg16-397923af.pth 这个模型是不带bn的VGG16
import torchvision
import torchvision.models as models
import torch
from torch import optim
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def set_trainable(model, class_num):
    for para in model.parameters():
        para.requires_grad = False

    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, class_num))

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()  # train 模式 启用 BatchNormalization 和 Dropout
    loss_curve = []
    accuracy_curve = []
    for batch_idx, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)  # 把tensor送到相应设备上
        optimizer.zero_grad()  # 梯度清零
        output = model(img)  # 前向推理，自动调用model.forward
        # shape:   output [128,2] , label [128]
        loss = criterion(output, label)  # 计算指定损失函数 注意CrossEntropyLoss()的target输入是类别值，不是one-hot编码格式
        loss.backward()  # 有了loss之后反向传播
        optimizer.step()  # 更新参数
        print('loss = ', loss.item())
        loss_curve.append(loss.item())
        pred = output.max(1, keepdim=False)[1]  # [0]是最大值 [1]是下标  找到概率最大的下标 就是预测值 0是每列的最大值，1是每行的最大值

        print('pred = ', pred)
        print('label = ', label)
        # correct1 = pred.eq(label.view_as(pred)).sum().item() #如果max参数 keepdim的话需要
        correct = (pred == label).sum().item()
        # print('correct = ', correct1, correct2)
        acc = correct / BATCH_SIZE
        accuracy_curve.append(acc)
        # 记录数据




model = models.vgg16_bn(pretrained=True) # 在vgg模块里找下载在本地的文件和model对应vgg的关系 有多仲vgg 不对应好就得下载
#print(model)
MODEL_PATH = './DNN/models/VGG16_bn.pth'
TRAIN_PATH = './dataset/cats_and_dogs/train'
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),#  The ToTensor transform should come before the Normalize transform, since the latter expects a tensor, but the Resize transform returns an image.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # 可以选做数据增强
    # transforms.RandomRotation(20),
    # transforms.RandomHorizontalFlip(),
])


# 读取训练集
train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_PATH,
                                                 transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=0,
                          shuffle=True)

#print(train_dataset.class_to_idx)
#print(train_dataset.imgs)



set_trainable(model, 2) # 0 1猫狗二分类
model = model.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters()) #使用adam优化器 只优化新建层的参数
print(model)

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch, criterion)
    torch.save(model.state_dict(), MODEL_PATH)  # 只存字典，直接输入model的话会保存整个模型






