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

from cnn_model import CGP2CNN
from cnn_model import CGP2CNN_autoencoder
from cnn_model import CGP2CNN_autoencoder_full
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
    # print(classname)
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
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
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
        self.imgSize = 160

        # load dataset
        if dataset_name == 'cifar10' or dataset_name == 'bsds' or dataset_name == 'mnist':
            if dataset_name == 'cifar10':
                self.n_class = 10
                self.channel = 3
                self.pad_size = 4
                # self.imgSize = 32
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
                # train, test = chainer.datasets.get_mnist(withlabel=True, ndim=3, scale=1.0)
            elif dataset_name == 'bsds':    # mnist
                self.n_class = 10
                self.channel = 1
                self.pad_size = 4
                data_transform = transforms.Compose([transforms.ToTensor()])
                dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/gray/re', transform=data_transform)
                # dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/re')
                self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=int(2))
                test_dataset = dset.ImageFolder(root='/home/suganuma/dataset/BSR/BSDS500/data/gray/re_v', transform=data_transform)
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=int(2))
            # model validation mode
            if validation:
                pass
            else:
                pass
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    def __call__(self, cgp, gpuID, epoch_num=200, batchsize=64, weight_decay=1e-4, eval_epoch_num=10,
                 data_aug=True, comp_graph='comp_graph.dot', out_model='mymodel.model', init_model=None,
                 retrain_mode=False):
        if self.verbose:
            print('GPUID    :', gpuID)
            print('epoch_num:', epoch_num)
        
        torch.backends.cudnn.benchmark = True # 画像サイズが変わらないときは高速?

        # model = CGP2CNN_autoencoder(cgp, self.channel, self.n_class, self.imgSize)
        model = CGP2CNN_autoencoder_primitive(cgp, self.channel, self.n_class, self.imgSize)
        init_weights(model, 'normal')
        model.cuda(gpuID)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
        
        # if init_model is not None:
        #     if self.verbose:
        #         print('\tLoad model from', init_model)
        #     serializers.load_npz(init_model, model)

        eval_epoch_num = np.min((eval_epoch_num, epoch_num))
        test_accuracies = np.zeros(eval_epoch_num)

        input = torch.FloatTensor(batchsize, self.channel, self.imgSize, self.imgSize)
        input = input.cuda(gpuID)
        input2 = torch.FloatTensor(batchsize, self.channel, self.imgSize, self.imgSize)
        input2 = input2.cuda(gpuID)
        for epoch in range(1, epoch_num+1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            ite = 0
            ave_psnr = 0
            for _, (data, target) in enumerate(self.dataloader):
                data = data[:,0:1,:,:]
                # data = data.contiguous().view(data.size(0),-1) # fullの場合
                data, target = data.cuda(gpuID), target.cuda(gpuID)
                for std in range(10,101,10):           
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
                    if ite == 0 and std == 40:
                        vutils.save_image(data_noise.data, './noise_samples%d.png' % gpuID, normalize=False)
                        vutils.save_image(input2_.data, './org_samples%d.png' % gpuID, normalize=False)
                        vutils.save_image(output.data, './output%d.png' % gpuID, normalize=False)
                ite += 1
            train_loss /= len(self.dataloader.dataset)
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('time ', time.time()-start_time)
            if epoch % 5 == 0:
                t_loss = self.__test(model, criterion, gpuID, input, input2)
        
        torch.save(model.state_dict(), './model_%d.pth' % (gpuID))
        return t_loss


    def __test(self, model, criterion, gpuID, input, input2):
        # model.eval()
        test_loss = 0
        ave_psnr = 0
        count = 0
        # for data, target in test_loader:
        for _, (data, target) in enumerate(self.test_dataloader):
            data = data[:,0:1,:,:]
            data, target = data.cuda(gpuID), target.cuda(gpuID)
            for std in range(10,101,10):           
                input.resize_as_(data).copy_(data)
                input_ = Variable(input, volatile=True)
                input2.resize_as_(data).copy_(data)
                input2_ = Variable(input2)
                data_noise = self.gaussian_noise(input_, 0.0, std)
                try:
                    output = model(data_noise, None)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(output, input2_)
                test_loss += loss.data[0]
                psnr = -10 * math.log10(loss.data[0])
                ave_psnr += psnr
                count += 1
                if std == 40:
                    vutils.save_image(data_noise.data, './test_noise_samples.png', normalize=False)
                    vutils.save_image(output.data, './test_output.png', normalize=False)

        test_loss /= len(self.test_dataloader.dataset)
        ave_psnr /= count
        print('Test set: loss: {:.4f}'.format(test_loss))
        print('Test set: PSNR: {:.4f}'.format(ave_psnr))
        return ave_psnr

    def gaussian_noise(self, inp, mean, std):
        noise = Variable(inp.data.new(inp.size()).normal_(mean, std))
        noise = torch.div(noise, 255.0)
        return inp + noise