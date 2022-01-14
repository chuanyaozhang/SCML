from __future__ import print_function
import sys
import argparse
import time
import math
import torch
import torch.nn as nn
#import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
import os
import glob
from itertools import chain
import numpy as np
import torch.optim as optim
from networks.resnet_big import SupConResNet
from torchvision import transforms, datasets
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
import random
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]='6,7'

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


class CNNmodel(nn.Module):
    def __init__(self,opt,inchannel=3, outchannel=64, stride=2):
        super(CNNmodel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,kernel_size=(3,3),stride=(stride,stride), padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel, kernel_size=(3,3), stride=(2,2), padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2,stride=2),
            # nn.Dropout(0.25),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3,3), stride=(2,2), padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3,3) ,stride=(2,2), padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
       
        self.head = nn.Sequential(
        nn.Linear(2304, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128)

        )

    def forward(self, x):
        opt = x.shape
        out = self.encoder(x)
        out = out.view(opt[0], -1)
        feat = F.normalize(self.head(out), dim=1)
        return feat

#################################

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')

    parser.add_argument('--save_log', type=bool, default=True,
                        help='save log')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    
   
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')

                    
    parser.add_argument('--n_way', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--k_shot', type=int, default=1,
                        help='Number of samples per category')
    parser.add_argument('--val_k_shot', type=int, default=10,
                        help='Number of val_samples per category')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)

    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')

    parser.add_argument('--momentum', type=float, default=0.9,help='momentum')




    # model dataset
    parser.add_argument('--model', type=str, default='resnet18',help='encoder model ,chose from resnet18,CNNmodel')
    parser.add_argument('--dataset', type=str, default="mini_imagenet",
                        choices=['mini_imagenet', 'AudioMnist-master-60','ESC-50'], help='dataset')

    parser.add_argument('--ckpt', type=str, default='./save/SimCLR/mini_imagenet_models/SimCLR_mini_imagenet_resnet18_lr_0.05_decay_0.0001_bsz_32_temp_0.07_trial_0_task_100/ckpt_epoch_255.pth',
                         help='path to pre-trained model')
                                                        
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=str, default='40,60,80,100,120',
                        help='where to decay lr, can be a list')
    opt = parser.parse_args()
     
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def loadDataPath(path, n_way, k_shot, val_k_shot, format='jpg'):#每次随机选取五类，每类一张n张图片
   
    n =n_way
    k =k_shot
    val_k=val_k_shot
    files = os.listdir(path)
    files = random.sample(files, n)
    support_paths = []
    support_lable=[]
    query_paths=[]
    query_lable=[]
  
    for lable,i in enumerate(files):
        #print(i)
        temp_data_path = glob.glob(path + "/" + i + '/*.' + format)
        #将每一类划分为训练集，验证集测试集

        random.shuffle(temp_data_path)
        support_path = temp_data_path[:400]
        query_path = temp_data_path[400:]

        support_set= random.sample(support_path, k)
        query_set = random.sample(query_path,val_k)
        

        support_paths.extend(support_set)
        support_lable.extend([lable for j in range(k)])###########生成类标签

        query_paths.extend(query_set)
        query_lable.extend([lable for j in range(val_k)])


    print(support_lable)
    return  support_paths,support_lable,query_paths,query_lable# 包含5类5张


class DataSet(Dataset):
    def __init__(self, X , lable,k_shot):
        self.lable=lable
        self.X = X
        self.K = k_shot
        self.transform=transforms.Compose([transforms.Resize(92),
                                          transforms.CenterCrop(84),
                                          transforms.ToTensor(),
                                          transforms.Normalize(np.array([0.485,0.456,0.406]),
                                          np.array([0.229,0.224,0.225]))])
    def __getitem__(self, index):
        x = self.X[index]
        x = Image.open(x)
        image=self.transform(x)
        target=self.lable[index]

        return image,target #存疑

    def __len__(self):
        return len(self.X)


def set_loader(opt):
    n_way=opt.n_way
    k_shot=opt.k_shot

    if opt.dataset =='mini_imagenet':
        train_data, train_lable, val_data, val_lable = loadDataPath("../1/test",n_way,k_shot,opt.val_k_shot,'jpg')#我需要构建tasks 每个tasks包含5类5shot图片，然后对训练的tasks验证训练结果
    
        train_dataset = DataSet(train_data,train_lable, opt.k_shot)
        val_dataset=DataSet(val_data, val_lable, opt.val_k_shot)
       

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5,shuffle=True)
        val_loader= torch.utils.data.DataLoader(val_dataset, batch_size=30,shuffle=True)#, shuffle=True
       
   
  
        #tain_dataset包含了所有的文件名称，我需要从中选择5类每类5张图片构成一个tasks，（可以先按类将文件选着好，然后将25个路径命名作为参数传给dataset,然后由dataset构建dataloader）
        #验证集，从训练集中选择相同的五类，任意8个元素，进行测试

    else:
        raise ValueError(opt.dataset)
    return train_loader, val_loader


def set_model(opt):
    # model = SupConResNet(name=opt.model)

    if opt.model=="CNNmodel":
        model=CNNmodel(opt)
        print("CNNmodel加载成功")
    else:
        model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    # if opt.model == 'CNNmodel':
    #     classifier = LinearClassifier(name=opt.model,dataset=opt.dataset)
    # else:
    #     classifier = LinearClassifier(name=opt.model,dataset=opt.dataset)

    ckpt = torch.load(opt.ckpt)
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:#多gpu模型加载到单gpu上
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        # classifier = classifier.cuda()
        criterion = criterion.cuda()
        model.load_state_dict(state_dict)
       # classifier.load_state_dict(ckpt["classifier"])
        print("CUDA模型加载成功")
    else:
        model.load_state_dict(state_dict)
       # classifier.load_state_dict(ckpt["classifier"])
        print("cpu模型加载成功")
    # for i in model.parameters():
    #     i.requires_grad=False
    return model,criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):#设计为k-way-n-shot的数据集，即每次输入5张不同种类的n张图片，训练模型。
    """one epoch training"""

    model.train()
    classifier.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            images = images.float().cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]


        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        features = model.encoder(images)
       
        output = classifier(features.detach())#stop gradent
        #output = classifier(features)
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
        #     print('Train: [{0}][{1}/{2}] '
        #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f}) '
        #           'DT {data_time.val:.3f} ({data_time.avg:.3f})'
        #           'loss {loss.val:.3f} ({loss.avg:.3f})'
        #           'train_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))
        #     sys.stdout.flush()
    
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):#使用qurey集验证准确度
    """validation"""
    model.eval()
    classifier.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
           # print("testloade")
            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:#.val即为当前batch的准确度
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'test_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #            idx, len(val_loader), batch_time=batch_time,
            #            loss=losses, top1=top1))

   # print('* test_Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

class down_task_liner(nn.Module):
    """Linear classifier"""
    def __init__(self, name='CNNmodel', num_classes=5, dataset="mini_imagenet",feat_dim=5120):
        super(down_task_liner, self).__init__()
        if dataset=="mini_imagenet":
            if name=='CNNmodel':
                feat_dim=2304
            elif  name=='resnet12':
                feat_dim=16000
            elif  name=='resnet18':
                feat_dim=512
            elif  name=='resnet34':
                feat_dim=512
        self.fc=nn.Sequential(nn.Linear(feat_dim, 512, bias=False), nn.BatchNorm1d(512),nn.ReLU(inplace=True), nn.Linear(512,num_classes, bias=True))
        #self.fc=nn.Sequential(nn.Linear(feat_dim,num_classes, bias=True))
    def forward(self, features):
        features = features.view(features.size(0), -1)
        return self.fc(features)

def main():
    os.environ['PYTHONHASHSEED']=str(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed(3)
    torch.cuda.manual_seed_all(2)
    np.random.seed(4)
    random.seed(6)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=True
    opt = parse_option()
    all_test=[]
    for seed in range(3):
        random.seed(seed)
        sum_test=[]#保存十次结果
    
        for i in range(0,30):
            # build data loader
            #random.seed(i)
            train_loader, val_loader = set_loader(opt)
            # build model and criterion
            model,criterion = set_model(opt)
            classifier=down_task_liner(name=opt.model,dataset=opt.dataset,num_classes=opt.n_way)
        
            if torch.cuda.is_available():
                classifier=classifier.cuda()
            best_acc = 0
            best_epoch=0
            optimizer= optim.SGD(params=chain(model.parameters(),classifier.parameters()),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
                # training routine
            for epoch in range(1, opt.epochs + 1):
                adjust_learning_rate(opt, optimizer, epoch)
                # train for one epoch
                time1 = time.time()
                train_loss, train_acc = train(train_loader, model, classifier, criterion,optimizer, epoch, opt)
                time2 = time.time()
                print('Tasks {}, Train epoch {}, total time {:.2f}, train_loss:{:.2f} , accuracy:{:.2f}'.format(i, epoch,time2 - time1,train_loss, train_acc))
                # eval for one epoch
                val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
                print('Tasks {}, test  epoch {}, total time {:.2f}, test_loss :{:.2f} , accuracy:{:.2f}'.format(i,epoch ,time2 - time1,val_loss, val_acc))
                # test_loss, test_acc = validate(test_loader, model, classifier, criterion, opt)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch= epoch
                ##保存验证效果最好的模型####
                    # save_file=os.path.join('./save_train',opt.dataset)
                    # if not os.path.isdir(save_file):
                    #     os.makedirs(save_file)
                    # save_file = os.path.join(save_file,'ckpt_epoch_{epoch}.pth'.format(epoch=best_epoch))
                    # state = {'model': model.state_dict(), 'classifier': classifier.state_dict(), 'epoch': epoch}
                    # torch.save(state,save_file)
            print("best_val_acc,best_epoch=",best_acc,best_epoch)
            best_acc=best_acc.tolist()
            sum_test.append(best_acc)
            print("sum_test:",sum_test)
            print("mean_acc:{},var_acc{}".format(np.mean(sum_test),np.var(sum_test)))


            # ckpt = torch.load(save_file)
            # model.load_state_dict( ckpt['model'])
            # classifier.load_state_dict( ckpt['classifier'])
            # All_test_acc=[]

            # test_loss, test_acc = validate(test_loader, model, classifier, criterion, opt)
            # print(test_acc)
            # test_acc=test_acc.tolist()
            # sum_test.append(test_acc)

        print(sum_test)
        print(np.mean(sum_test),np.var(sum_test))
        all_test.append(np.mean(sum_test))
    print(all_test,np.mean(all_test),np.var(all_test))

if __name__ == '__main__':
    main()
