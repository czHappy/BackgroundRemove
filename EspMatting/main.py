'''
Author  : Zhengwei Li
Version : 1.0.0 
'''
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import data
from model import segnet
import DataLoaderX
import pdb
import data_prefetcher
def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Fast portrait matting !')
    parser.add_argument('--dataDir', default='./DATA/', help='dataset directory')
    parser.add_argument('--saveDir', default='./result', help='save result')
    parser.add_argument('--trainData', default='portrait_matting', help='train dataset name')
    parser.add_argument('--trainList', default='./DATA/list.txt', help='train img ID')
    parser.add_argument('--load', default= 'FPM', help='save model')

    parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')

    parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')
    parser.add_argument('--train_refine', action='store_true', default=False, help='train refine')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=8, help='input batch size for train')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for train')


    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')

    args = parser.parse_args()
    print(args)
    return args


def set_lr(args, epoch, optimizer):

    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2**epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

  

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.load)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
            
    def save_model(self, model, epoch):

        # epoch_out_path = "{}/ckpt_e{}.pth".format(self.save_dir_model, epoch)
        # print("Checkpoint saved to {}".format(epoch_out_path))

        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        # }, epoch_out_path)

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model    


    def save_log(self, log):
        self.logFile.write(log + '\n')


def fusion_loss(args, img, mask_gt, seg, alpha_gt, alpha, eps=1e-6):

    #print('seg type : ', seg.dtype)
    #print('mask_gt type : ', mask_gt.dtype)
    #print('tensor.long() type : ', (mask_gt[:, 0, :, :].long()).dtype)
    criterion = nn.CrossEntropyLoss()
    #np.set_printoptions(threshold=sys.maxsize)
    #print(mask_gt[:, 0, :, :].long().numpy())
    # 计算网络输出seg和mask_gt的交叉熵损失 注意这里是将mask_gt取单通道
    # 注意，seg从模型输出来看是2通道，这里交叉熵的label参数是单通道，原因是pytorch交叉熵的label不需要one-hot格式
    cross_entropy_loss = criterion(seg, mask_gt[:, 0, :, :].long()) # 将target Tensor进行类型转换 torch.int64; 因为target参数应该是整数 torch中的int默认是int64

    # paper loss : Fast Deep Matting for Portrait Animation on Mobile Phone

    # 计算网络输出的alpha和alpha_GT之间的损失
    L_alpha = torch.sqrt(torch.pow(alpha_gt - alpha, 2.) + eps).mean()
    # 即concat, 按维度1拼接，实际上是扩展为3通道
    # 算术乘，相当于用掩码过滤图片背景
    gt_msk_img = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img

    alpha_img = torch.cat((alpha, alpha, alpha), 1) * img

    # 这里计算去背景得到图片的色彩差损失
    L_color = torch.sqrt(torch.pow(gt_msk_img - alpha_img, 2.) + eps).mean()

    if args.train_refine:
        return L_alpha + L_color, L_alpha, L_color, cross_entropy_loss
    else:
        return L_alpha + L_color + cross_entropy_loss, L_alpha, L_color, cross_entropy_loss



def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is is available !")
    print('Device: ', device)
    print('===> Building model ...')
    model = segnet.SegMattingNet()    

    model.to(device)

    print('===> Loading datasets ...')
    # 获得data类中的trainData字符串指明的属性名，传入dataDIr, trainList, pathch_size等参数
    # 有点像java反射 动态获取方法
    # 返回一个自制数据集，DataLoader读取
    # Batch是批量的大小，就是训练的时候每次输入多少张图片到NN。
    # Patch是图像块的大小，比如说原图1024*1024，随机从图中裁剪出256*256大小的块，就是patch。

    train_data = getattr(data, args.trainData)(base_dir = args.dataDir, \
                                                  imglist = args.trainList, \
                                                  patch = args.patch_size)

    #trainloader = DataLoaderX.DataLoaderX(train_data, batch_size=args.train_batch, drop_last=True, shuffle=True, num_workers=args.nThreads, pin_memory=True)
    trainloader = DataLoader(train_data, batch_size=args.train_batch, drop_last=True, shuffle=True, num_workers=args.nThreads, pin_memory=True)
    print('===> Set optimizer ...')

    lr = args.lr
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
    #                                lr=lr, momentum=0.99, weight_decay=0.0005)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                   lr=lr, betas=(0.9, 0.999), weight_decay=0.0005, amsgrad=False)    

    print("===> Start Train ! ... ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning: #如果是finetuning，那就微调模型，先读取上次训练的模型
        start_epoch, model = trainlog.load_model(model) 

    model.train() #设定模型为train模式
    for epoch in range(start_epoch, args.nEpochs+1):

        loss_tr = 0
        loss_ = 0
        L_alpha_ = 0
        L_color_ = 0
        L_cross_ = 0
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)

        #total = 0
        t0 = time.time()
        prefetcher = data_prefetcher.data_prefetcher(trainloader)
        batch = prefetcher.next()
        i = 0
        while batch is not None:
            print('i = ', i, len(batch[0]))
            img, mask_gt, alpha_gt = batch[0], batch[1], batch[2]
            img, mask_gt, alpha_gt = img.to(device), mask_gt.to(device), alpha_gt.to(device)

            seg, alpha = model(img)
            loss, L_alpha, L_color, L_cross = fusion_loss(args, img, mask_gt, seg, alpha_gt, alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ += loss.item()
            L_alpha_ += L_alpha.item()
            L_color_ += L_color.item()
            L_cross_ += L_cross.item()
            print('batch {} complete! loss = {}'.format(i, loss))
            if (i+1) % 10 == 0:
                t1 = time.time()
                print('10 batch cost time: ', t1 - t0)

            i += 1
            batch = prefetcher.next()
        if epoch % args.save_epoch == 0:
            print('236 i = ', i)
            #print('236 total = ', total)
            loss_ = loss_ / (i+1)
            L_alpha_ = L_alpha_ / (i+1)
            L_color_ = L_color_ / (i+1)
            L_cross_ = L_cross_ / (i+1)

            log = "[{} / {}] \tLr: {:.5f}\nloss: {:.5f}   loss_alpha: {:.5f}   loss_color: {:.5f}   loss_cross: {:.5f}"\
            .format(epoch, args.nEpochs,
                lr, loss_, L_alpha_, L_color_, L_cross_)
            print(log)
            trainlog.save_log(log)
            trainlog.save_model(model, epoch)


if __name__ == "__main__":
    main()
