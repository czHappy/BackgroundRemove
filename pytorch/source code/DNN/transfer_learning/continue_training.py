import torchvision
import torchvision.models as models
import torch
from torch import optim
import torch.nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import cv2 as cv
import argparse

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='VGG16 classfier for cats and dogs!')
    parser.add_argument('--continue_training', type=bool, default=False, help='continue to train from the last_ckpt')

    args = parser.parse_args()
    print(args)
    return args


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

'''
def model_save(model, optimizer, epoch, save_dir):
    ckpt = {'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch}
    path = os.path.join(save_dir, 'VGG16_bn_{}_epoch.pth'.format(epoch))
    torch.save(ckpt, path)
'''
def model_save(model, optimizer, epoch, save_dir):
    ckpt = {'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch}
    path = os.path.join(save_dir, 'VGG16_bn_continue.pth')
    torch.save(ckpt, path)


def model_load(model, optimizer, save_dir):
    ckpt = torch.load(os.path.join(save_dir, 'VGG16_bn_continue.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    return epoch



def image_show_sample(dir):
    img = os.listdir(dir)[0]
    print(os.path.join(dir, img))
    img = cv.imread(os.path.join(dir, img))

    cv.imshow('img', img)
    cv.waitKey(0)


args = get_args()
model = models.vgg16_bn(pretrained=args.continue_training) # 在vgg模块里找下载在本地的文件和model对应vgg的关系 有多仲vgg 不对应好就得下载
#print(model)
MODEL_DIR = './DNN/models/'
TRAIN_PATH = './dataset/cats_and_dogs/train'
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


#image_show_sample(os.path.join(TRAIN_PATH, 'cat'))

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
#print(model.state_dict())
#print('opt lr1 = ', optimizer.param_groups[0]['lr'])
if args.continue_training:
    start_epoch = model_load(model, optimizer, MODEL_DIR)

print(start_epoch)
#print('opt lr2 = ', optimizer.param_groups[0]['lr'])
#print(model.state_dict())

for epoch in range(start_epoch+1, start_epoch + EPOCHS+1):
    train(model, DEVICE, train_loader, optimizer, epoch, criterion)
    model_save(model, optimizer, epoch, MODEL_DIR)






