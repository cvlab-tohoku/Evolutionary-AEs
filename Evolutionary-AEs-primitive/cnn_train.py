#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random

from cnn_model import CGP2CNN
from cnn_model import CGP2CNN_autoencoder
from cnn_model import CGP2CNN_autoencoder_primitive


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
        # init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class basicNet(nn.Module):
    def __init__(self):
        super(basicNet, self).__init__()
        input_channel = 3
        filters = [16,32,64,128,256,512]
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, filters[2], 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], 1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], 1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], 1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[2], filters[2], 1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filters[2], input_channel, 3, stride=1, padding=1, bias=False),
        )

    def forward(self, input, t):
        out = self.main(input)
        return out


# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_name, validation=True, valid_data_ratio=0.1, verbose=True, imgSize=32):
        # dataset_name: name of data set ('cifar10' or 'cifar100' or 'mnist')
        # validation: [True]  model validation mode
        #                     (split training data set according to valid_data_ratio for evaluation of CGP individual)
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # valid_data_ratio: ratio of the validation data
        #                    (e.g., if the number of all training data=50000 and valid_data_ratio=0.2, 
        #                       the number of training data=40000, validation=10000)
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = 64
        self.validation = validation

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'bsds' or dataset_name == 'mnist':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                self.pad_size = 4
                dataset = dset.CIFAR10(root='./', download=True,
                           transform=transforms.Compose([
                               transforms.Scale(self.imgSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=int(1))
            elif dataset_name == 'mnist':    # mnist
                self.n_class = 10
                self.channel = 1
                self.pad_size = 4
                # self.imgSize = 32
                dataset = dset.MNIST(root='./', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(self.imgSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ]))
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=int(2))
                test_dataset = dset.MNIST(root='./', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(self.imgSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                           ]))
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=int(2))
            elif dataset_name == 'bsds':
                if self.validation:
                    self.n_class = 10
                    self.channel = 3
                    self.pad_size = 4
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(64, 0), transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/color/color/train', transform=data_transform)
                    self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=int(2))
                    test_dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/color/color/val', transform=test_data_transform)
                    self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(2))
                else:
                    self.n_class = 10
                    self.channel = 3
                    self.pad_size = 4
                    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(64, 0), transforms.ToTensor()])
                    test_data_transform = transforms.Compose([transforms.ToTensor()])
                    dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/color/retrain/train', transform=data_transform)
                    self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=int(2))
                    test_dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/color/retrain/test', transform=test_data_transform)
                    self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(2))
                print('train num', len(self.dataloader.dataset))
                print('test num ', len(self.test_dataloader.dataset))
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=200, batchsize=64, weight_decay=1e-4, eval_epoch_num=10,
                 data_aug=True, comp_graph='comp_graph.dot', out_model='mymodel.model', init_model=None,
                 retrain_mode=False):
        if self.verbose:
            print('GPUID    :', gpuID)
            print('epoch_num:', epoch_num)
        
        torch.backends.cudnn.benchmark = True
        model = CGP2CNN_autoencoder_primitive(cgp, self.channel, self.n_class, self.imgSize)
        # init_weights(model, 'normal')
        model.cuda(gpuID)
        criterion = nn.MSELoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)

        eval_epoch_num = np.min((eval_epoch_num, epoch_num))
        test_accuracies = np.zeros(eval_epoch_num)

        input = torch.FloatTensor(batchsize, self.channel, self.imgSize, self.imgSize)
        input = input.cuda(gpuID)
        input2 = torch.FloatTensor(batchsize, self.channel, self.imgSize, self.imgSize)
        input2 = input2.cuda(gpuID)

        std = 20

        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            ite = 0
            ave_psnr = 0
            for module in model.children():
                module.train(True)
            for _, (data, target) in enumerate(self.dataloader):
                # data = data[:,0:1,:,:]  # in the case of using gray-scale images(bacause dataloader gives 3-dimension batches even if we input gray-scale images)
                data, target = data.cuda(gpuID), target.cuda(gpuID)
                for _ in range(1,50,1):
                    input.resize_as_(data).copy_(data)
                    input_ = Variable(input)
                    input2.resize_as_(data).copy_(data)
                    input2_ = Variable(input2)
                    data_noise = self.gaussian_noise(input_, 0.0, std)
                    optimizer.zero_grad()
                    try:
                        output = model(data_noise, None)
                    except:
                        import traceback
                        traceback.print_exc()
                        return 0.
                    loss = criterion(output, input2_)
                    train_loss += loss.data[0]
                    loss.backward()
                    optimizer.step()
                    if ite == 0:
                        vutils.save_image(data_noise.data, './noise_samples%d.png' % gpuID, normalize=False)
                        vutils.save_image(input2_.data, './org_samples%d.png' % gpuID, normalize=False)
                        vutils.save_image(output.data, './output%d.png' % gpuID, normalize=False)
                ite += 1
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time()-start_time)
            if epoch % 10 == 0:
                for module in model.children():
                    module.train(False)
                t_loss = self.__test_per_std(model, criterion, gpuID, input, input2, std)
            if epoch == 200:
                for param_group in optimizer.param_groups:
                    tmp = param_group['lr']
                tmp *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tmp
            if epoch == 400:
                for param_group in optimizer.param_groups:
                    tmp = param_group['lr']
                tmp *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tmp
        
        torch.save(model.state_dict(), './model_%d.pth' % int(gpuID))
        return t_loss

    def gaussian_noise(self, inp, mean, std):
        noise = Variable(inp.data.new(inp.size()).normal_(mean, std))
        noise = torch.div(noise, 255.0)
        return inp + noise

    def __test_per_std(self, model, criterion, gpuID, input, input2, std):
        test_loss = 0
        ave_psnr = 0
        count = 0
        print('std', std)
        count = 0
        for _, (data, target) in enumerate(self.test_dataloader):
            # data = data[:,0:1,:,:]
            data, target = data.cuda(gpuID), target.cuda(gpuID)
            input.resize_as_(data).copy_(data)
            input_ = Variable(input, volatile=True)
            input2.resize_as_(data).copy_(data)
            input2_ = Variable(input2, volatile=True)
            data_noise = self.gaussian_noise(input_, 0.0, std)
            try:
                output = model(data_noise, None)
            except:
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(torch.mul(output, 255.0), torch.mul(input2_, 255.0))
            psnr = 10*math.log10(255*255/loss.data[0])
            # loss = criterion(output, input2_)
            # psnr = -10 * math.log10(loss.data[0])
            test_loss += loss.data[0]
            ave_psnr += psnr
            count += 1
        ave_psnr /= (count)
        test_loss /= (count)
        print('Test PSNR: {:.4f}'.format(ave_psnr))
        print('Test loss: {:.4f}'.format(test_loss))
        vutils.save_image(data_noise.data, './test_noise_samples_std%02d.png' % int(std), normalize=False)
        vutils.save_image(output.data, './test_output_std%02d.png' % int(std), normalize=False)
        vutils.save_image(input2_.data, './test_org_std%02d.png' % int(std), normalize=False)

        test_loss /= len(self.test_dataloader.dataset)
        return ave_psnr
