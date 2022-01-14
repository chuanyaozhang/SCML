from __future__ import print_function

import os
import sys
import argparse
import time
import math
import glob

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from torchvision import transforms, datasets
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate,accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from torch.utils.data import Dataset, DataLoader
from PIL  import Image
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
os.environ["CUDA_VISIBLE_DEVICES"]='3,4'
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size,default=256')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use ,16')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    #few shot setting
    parser.add_argument('-its', '--iterations', type=int, help='number of episodes per epoch, default=100',  # tasks
                        default=500)  # tasks的数量，相当于有多少个batch

    parser.add_argument('-cTr', '--classes_per_it_tr',type=int,
                        default=5,
                        help='n way for train, default=60')
                        
    parser.add_argument('-nsTr', '--num_support_tr',type=int, help='n shot for train suport, default=5', 
                        default=15)

    parser.add_argument('-nqTr', '--num_query_tr',type=int,help='n shot for train query, default=5',
                        default=10)

    parser.add_argument('-cVa', '--classes_per_it_val',type=int,help='n way for val, default=5', 
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val', type=int, help=' n shot for val suport,default=5', 
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val', type=int, help='n shot for val query, default=15',
                        default=15)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet12',help='resnet18,resnet50')
    parser.add_argument('--dataset', type=str, default='mini_imagenet',
                        choices=['cifar10', 'cifar100', 'path','mini_imagenet'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=84, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save_our/{}/{}_models'.format(opt.method,opt.dataset)
    opt.tb_path = './save_our/{}/{}_tensorboard'.format(opt.method,opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_task_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial,opt.iterations)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def loadDataPath(path,format='jpg'):
    files = os.listdir(path)
    data_paths = []
    label=[]
    j=0
    for i in files:
        data_path = glob.glob(path+ "/" + i + '/*.' + format)
        data_paths.extend(data_path)
        label.extend([j for _ in data_path])
        j=j+1
    return data_paths , label

class PrototypicalBatchSampler(object):
    def __init__(self, labels, classes_per_it, suport_num_samples,query_num_samples, iterations):
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it # n_way
        self.sample_suport_per_class = suport_num_samples
        self.sample_query_per_class = query_num_samples
        self.iterations = iterations
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
    def __iter__(self):
        for it in range(self.iterations):
            class_sample=random.sample(list(self.classes), k=self.classes_per_it)#不放回选取，choice为放回选取

            batch=[]
            for i in range(self.sample_suport_per_class+self.sample_query_per_class):
                for class_temp in class_sample:
                    #先找到某一类的全部索引，然后在索引中选取1个
                    class_index=[i for i,x in enumerate(self.labels) if x == class_temp]
                    one_class_index=random.sample([ i for i in class_index if i not in batch ],k=1)#求列表差集
                    batch.extend(one_class_index)
            #print(np.unique([self.labels[i] for i in batch], return_counts=True))
            yield batch
    def __len__(self):
        return self.iterations


def init_sampler(opt, labels, mode):#label为一个标签列表  2021.8.11日添加
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr #number of random classes per episode for training, default=5
        num_samples = opt.num_support_tr + opt.num_query_tr#number of samples per class to use as support/query for training, default=5+5
    else:
        classes_per_it = opt.classes_per_it_val#number of random classes per episode for validation, default=5
        num_samples = opt.num_support_val + opt.num_query_val#number of samples per class to use as support/query for validation, default=5+15

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    suport_num_samples= opt.num_support_tr,
                                    query_num_samples=opt.num_query_tr,
                                    iterations=opt.iterations)  #iterations number of episodes per epoch, default=100,num_samples 一次只返回一个batch的indices（索引)

class DataSet(Dataset):
    def __init__(self, X,label,transform):

        self.X = X 
        self.y = label
        self.transform_1=transform
    def __getitem__(self, index):

        x = self.X[index]
        x = Image.open(x)

        image_1=self.transform_1(x)
        image_2=self.transform_1(x)
        target= self.y[index]
        

        return [image_1,image_2],target#存疑
       
    
    def __len__(self):
        return len(self.X)

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset=="mini_imagenet":
        mean = (0.485,0.456,0.406)
        std = (0.229,0.224,0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset=="mini_imagenet":
        train_data,train_label = loadDataPath("../1/train",'jpg')
        train_dataset=DataSet(train_data,train_label,transform=train_transform)
        train_sampler = init_sampler(opt, train_dataset.y, "train")  # 采样生成一个tasks的索引，根据n-way-n-shot
        train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_sampler=train_sampler)
       # train_sampler = None
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader

    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer1,optimizer2,classifier,epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    p=opt.num_support_tr*opt.classes_per_it_tr
    end = time.time()
    for idx, (images, label_all) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images[0]=images[0].float()
        images[1] = images[1].type_as(images[0])

        images_suport_0 = images[0][:p]
        images_suport_1 = images[1][:p]
        labels = label_all[:p]

        images_query_0 = images[0][p:]
        images_query_1 = images[1][p:]
        labels_query = label_all[p:]

        images = torch.cat([images_suport_0, images_suport_1], dim=0)



       # images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer1)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

        if torch.cuda.is_available():
            images_query_0 = images_query_0.cuda(non_blocking=True)
            images_query_1 = images_query_1 .cuda(non_blocking=True)
            labels_query = labels_query .cuda(non_blocking=True)
            classifier =classifier.cuda()
        bsz = labels_query.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # with torch.no_grad():
        features = model.encoder(images_query_0)
        output = classifier(features)
        loss_CE=nn.CrossEntropyLoss()
        loss2 = loss_CE(output, labels_query)

        #loss2.update(loss2.item(), bsz)
        acc1, acc5 = accuracy(output, labels_query ,topk=(1, 5))
        print("acc1,acc5=",acc1,acc5)
        # SGD
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
    return losses.avg

def init_randomseed(seed):
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)
    random.seed(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=True

def print_log(log_name, str, save_log=True, print_time=True):
    if print_time:
        localtime = time.asctime(time.localtime(time.time()))
        str = "[ " + localtime + ' ] ' + str
    print(str)
    if (os.path.exists(log_name)==False):
        os.makedirs(log_name)
    if save_log:
        with open(log_name+'/train_loss.txt', 'a') as f:
            print(str, file=f)   
def main():

    init_randomseed(2)
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
    if opt.model=="resnet12":
        dim_in=16000
    elif opt.model=="resnet18":
        dim_in=512
    elif opt.model=="resnet34":   
        dim_in=512
    elif opt.model=="resnet50":   
        dim_in=2048
    classifier = nn.Sequential(nn.Linear(dim_in, dim_in, bias=False), nn.BatchNorm1d(dim_in),nn.ReLU(inplace=True), nn.Linear(dim_in,64, bias=True))

    # build optimizer
    optimizer = set_optimizer(opt, model)
    optimizer2= optim.Adam(params=chain(model.parameters(),classifier.parameters()),lr=0.0001) 

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    log_name = './save_our/{}/{}/{}_ktask{}_kway{}_ksuport{}_our'.format(opt.dataset,opt.method,opt.model,opt.iterations,opt.classes_per_it_tr,opt.num_support_tr)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, optimizer2,classifier,epoch,opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        str_train='epoch {}, total time {:.2f}, epoch  {},  loss  {}'.format(epoch, time2 - time1,epoch,loss)
        print_log(log_name, str_train, save_log=True, print_time=True)
        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
