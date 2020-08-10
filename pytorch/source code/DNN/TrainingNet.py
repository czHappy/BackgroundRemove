from DNN import ClassicalNet
from torch import optim
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
BATCH_SIZE = 256
LR = 0.02
EPOCHS = 30
MODEL_PATH = './models/AlexNet.pth'
TRAIN_PATH = './dataset/cats_and_dogs/train'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize(size=(227, 227)),
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
                          shuffle=True)

print(train_dataset.class_to_idx)
print(train_dataset.imgs)


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()  # train 模式 启用 BatchNormalization 和 Dropout

    for batch_idx, (img, label) in enumerate(train_loader):
        img, label = img.to(device), label.to(device)  # 把tensor送到相应设备上
        optimizer.zero_grad()  # 梯度清零
        output = model(img)  # 前向推理，自动调用model.forward
        # shape:   output [32,2] , label [32]
        loss = criterion(output, label)  # 计算指定损失函数 注意CrossEntropyLoss()的target输入是类别值，不是one-hot编码格式
        loss.backward()  # 有了loss之后反向传播
        optimizer.step()  # 更新参数
        print('loss = ', loss.item())
        # 记录数据


model = ClassicalNet.AlexNet().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()) #使用adam优化器

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch,criterion)

print(model)

#torch.save(model.state_dict(), MODEL_PATH) #只存字典，直接输入model的话会保存整个模型


