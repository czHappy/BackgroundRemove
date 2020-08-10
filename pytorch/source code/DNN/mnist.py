import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 512  # 批次大小
EPOCHS = 5  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
DOWNLOAD_MNIST = False

train_data = datasets.MNIST('../../../MNIST_data/', train=True, download=DOWNLOAD_MNIST,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

test_data = datasets.MNIST('../../../MNIST_data/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE, shuffle=True)

print("train_data:", train_data.data.size()) #torch.size() 查看tensor的维度
print("test_data:", test_data.data.size())


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1 ,10,5) # 24x24 input_channel output_channel kernel_size strides=1 (28-5+1) / 1 = 24
        self.pool = nn.MaxPool2d(2,2) # 12x12
        self.conv2=nn.Conv2d(10,20,3) # 10x10 (12-3+1)/1 = 10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0) #512,BATCH_SIZE
        #print(in_size)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = self.pool(out)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)# the size -1 is inferred from other dimensions, result:(BATCH_SIZE , N) 证明输入网络的数据以batch为单位
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()  # train 模式 启用 BatchNormalization 和 Dropout
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 把tensor送到相应设备上
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向推理，自动调用model.forward
        loss = F.nll_loss(output, target)  # 计算NLLLoss 负对数似然损失 根据标签把对应位置的值取出来去掉符号相加做平均 而交叉熵 = Softmax–Log–NLLLoss合并成一步
        loss.backward()  # 有了loss之后反向传播
        optimizer.step()  # 更新参数
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))  # 用item得到元素值


def test(model, device, test_loader):
    model.eval()  # eval 模式 关闭 BatchNormalization 和 Dropout
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters()) #使用adam优化器
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)