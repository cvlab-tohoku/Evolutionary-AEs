#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
from net_utils import conv2DBatchNorm, conv2DBatchNormRelu
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(0.2, inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_last(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_last, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.Tanh())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlockTranspose(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlockTranspose, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_cat(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_cat, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ConvBlock_sum(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ConvBlock_sum, self).__init__()
        pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding=pad_size, bias=False),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel):
        super(ResBlock, self).__init__()
        pad_size = kernel // 2
        self.convbnrelu1 = conv2DBatchNormRelu(in_size, out_size, kernel,  stride=1, padding=pad_size, bias=False)
        self.convbn2 = conv2DBatchNorm(out_size, out_size, kernel, stride=1, padding=pad_size, bias=False)
        # self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # residual = x
        # out = self.convbnrelu1(x)
        # out = self.convbn2(out)
        # out += residual
        # out = self.relu(out)
        # return out

        residual = x
        out = self.convbnrelu1(x) ###
        out = self.convbn2(out)
        in_data = [residual, out]
        # # check of the image size
        # small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
        # pool_num = xp.floor(xp.log2(in_data[large_in_id].size(2) / in_data[small_in_id].size(2)))
        # for _ in xp.arange(pool_num):
        #     in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
        

        # offset = outputs2.size()[2] - inputs1.size()[2]
        # padding = 2 * [offset // 2, offset // 2]
        # outputs1 = F.pad(inputs1, padding)
        # return self.conv(torch.cat([outputs1, outputs2], 1))

        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            # padding = [offset // 2, offset // 2]
            # tmp = Variable(in_data[large_ch_id].data.new(in_data[large_ch_id].size())
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return self.relu(out)

class MyMaxPooling(nn.Module):
    def __init__(self, pool_size, in_size):
        super(MyMaxPooling, self).__init__()
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        # self.conv = nn.Conv2d(in_size, in_size, 1)
        
    def forward(self, x):
        x = self.pool(x)
        # x = self.conv(x)
        return x

class MyAvgPooling(nn.Module):
    def __init__(self, pool_size, in_size):
        super(MyAvgPooling, self).__init__()
        self.pool = nn.AvgPool2d(pool_size, pool_size)
        # self.conv = nn.Conv2d(in_size, in_size, 1)
        
    def forward(self, x):
        x = self.pool(x)
        # x = self.conv(x)
        return x

class MyMaxUnPooling(nn.Module):
    def __init__(self, pool_size, in_size):
        super(MyMaxUnPooling, self).__init__()
        # self.unpool = nn.MaxUnpool2d(pool_size, pool_size)
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')
        # self.conv = nn.Conv2d(in_size, in_size, 1)

    def forward(self, x):
        x = self.unpool(x)
        # x = self.conv(x)
        return x



# Construct a CNN model using CGP (list)
class CGP2CNN(nn.Module):
    def __init__(self, cgp, in_channel, n_class, imgSize):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []
        self.decode = []
        self.channel_num = [None for _ in range(len(self.cgp))]
        self.size = [None for _ in range(len(self.cgp))]
        self.channel_num[0] = in_channel
        self.size[0] = imgSize
        i = 0
        # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
        for name, in1 in self.cgp:
            if name == 'pool_max':
                self.encode.append(nn.MaxPool2d(self.pool_size, self.pool_size))
                tmp = self.size[in1]
                tmp = int(tmp / 2)
                self.size[i] = tmp
                tmp = self.channel_num[in1]
                self.channel_num[i] = tmp
            elif name == 'pool_ave':
                self.encode.append(nn.AvgPool2d(self.pool_size, self.pool_size))
                tmp = self.size[in1]
                tmp = int(tmp / 2)
                self.size[i] = tmp
                tmp = self.channel_num[in1]
                self.channel_num[i] = tmp
            # elif name == 'concat':
            #     out_size = self.channel_num[in1]+self.channel_num[in2]
            #     self.encode.append(ConvBlock_cat(out_size, out_size, 3))
            #     self.channel_num[i] = out_size
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            # elif name == 'sum':
            #     # channel size check
            #     in_data = [self.channel_num[in1], self.channel_num[in2]]
            #     small_ch_id, large_ch_id = (0, 1) if in_data[0] < in_data[1] else (1, 0)
            #     self.encode.append(ConvBlock_sum(in_data[large_ch_id], in_data[large_ch_id], 3))
            #     self.channel_num[i] = in_data[large_ch_id]
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            elif name == 'ConvBlock1_1':
                out_size = 1
                # self.encode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock32_3':
                out_size = 32
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 3)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock32_5':
                out_size = 32
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 5)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock32_7':
                out_size = 32
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 7)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_3':
                out_size = 64
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 3)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_5':
                out_size = 64
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 5)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_7':
                out_size = 64
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 7)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_3':
                out_size = 128
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 3)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_5':
                out_size = 128
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 5)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))                
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_7':
                out_size = 128
                # self.arch[name] = ConvBlock(self.channel_num[in1], out_size, 7)
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock32_3':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 32, 3)
            elif name == 'ResBlock32_5':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 32, 5)
            elif name == 'ResBlock32_7':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 32, 7)
            elif name == 'ResBlock64_3':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 64, 3)
            elif name == 'ResBlock64_5':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 64, 5)
            elif name == 'ResBlock64_7':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 64, 7)
            elif name == 'ResBlock128_3':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 128, 3)
            elif name == 'ResBlock128_5':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 128, 5)
            elif name == 'ResBlock128_7':
                self.arch[name] = nn.Conv2d(self.channel_num[in1], 128, 7)
            elif name == 'full':
                input_num = int(self.size[in1] * self.size[in1] * self.channel_num[in1])
                # self.arch[name] = nn.Sequential(nn.Linear(input_num, n_class))
                # self.arch[name] = nn.Linear(input_num, n_class)
                self.encode.append(nn.Linear(input_num, n_class))
            i += 1
        # self.decode.reverse()
        # self.network = self.encode + self.decode
        self.layer_module = nn.ModuleList(self.encode)
        # self.model = nn.Sequential(self.arch)
        self.train = True
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp))]
        self.param_num = 0

    def main(self,x):
        out = x
        outputs = self.outputs
        outputs[0] = x    # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, torch.nn.modules.linear.Linear):
                tmp = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
                outputs[nodeID] = layer(tmp)
                # out = out.view(out.size(0), -1)
                # out = layer(out)
            elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                if outputs[self.cgp[nodeID][1]].size(2) > 1:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                else:
                    outputs[nodeID] = outputs[self.cgp[nodeID][1]]
            else:
                # print(type(layer))
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                # out = layer(out)
            nodeID += 1
        return outputs[-1]
        # return out

    def forward(self, x, t):
        return self.main(x)
        # # xp = chainer.cuda.get_array_module(x)
        # outputs = self.outputs
        # outputs[0] = x    # input image
        # nodeID = 1
        # param_num = 0
        # for name, f in self.arch.items():
        # # for f in range(1,len(self.temp)):
        # #     print(self.temp)
        # #     outputs[nodeID] = self.temp[f](outputs[self.cgp[nodeID][1]])
        # #     nodeID += 1


        #     if 'ConvBlock' in name:
        #         # outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]])
        #         outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         nodeID += 1
        #         # param_num += tmp_num
        #     elif 'ResBlock' in name:
        #         outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         # outputs[nodeID], tmp_num = getattr(self, name)(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][1]], self.train)
        #         nodeID += 1
        #         # param_num += tmp_num
        #     elif 'pool' in name:
        #         outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         nodeID += 1
        #         # check of the image size
        #         # if outputs[self.cgp[nodeID][1]].shape[2] > 1:
        #         #     outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         #     nodeID += 1
        #         # else:
        #         #     outputs[nodeID] = outputs[self.cgp[nodeID][1]]
        #         #     nodeID += 1
        #     elif 'concat' in name:
        #         outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         nodeID += 1
        #         # in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
        #         # # check of the image size
        #         # small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
        #         # pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
        #         # for _ in xp.arange(pool_num):
        #         #     in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
        #         # # concat
        #         # outputs[nodeID] = f(in_data[0], in_data[1])
        #         # nodeID += 1
        #     elif 'sum' in name:
        #         outputs[nodeID] = f(outputs[self.cgp[nodeID][1]])
        #         nodeID += 1
        #         # in_data = [outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]]]
        #         # # check of the image size
        #         # small_in_id, large_in_id = (0, 1) if in_data[0].shape[2] < in_data[1].shape[2] else (1, 0)
        #         # pool_num = xp.floor(xp.log2(in_data[large_in_id].shape[2] / in_data[small_in_id].shape[2]))
        #         # for _ in xp.arange(pool_num):
        #         #     in_data[large_in_id] = F.max_pooling_2d(in_data[large_in_id], self.pool_size, self.pool_size, 0, False)
        #         # # check of the channel size
        #         # small_ch_id, large_ch_id = (0, 1) if in_data[0].shape[1] < in_data[1].shape[1] else (1, 0)
        #         # pad_num = int(in_data[large_ch_id].shape[1] - in_data[small_ch_id].shape[1])
        #         # tmp = in_data[large_ch_id][:, :pad_num, :, :]
        #         # in_data[small_ch_id] = F.concat((in_data[small_ch_id], tmp * 0), axis=1)
        #         # # summation
        #         # outputs[nodeID] = in_data[0] + in_data[1]
        #         # nodeID += 1
        #     elif 'full' in name:
        #         y = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
        #         outputs[nodeID] = f(y)
        #         # nodeID += 1
        #         # outputs[nodeID] = getattr(self, name)(outputs[self.cgp[nodeID][1]])
        #         nodeID += 1
        #         # param_num += f.W.data.shape[0] * f.W.data.shape[1] + f.b.data.shape[0]
        #     else:
        #         print('not defined function at CGP2CNN __call__')
        #         exit(1)
        # # self.param_num = param_num

        # if t is not None:
        #     self.loss = None
        #     self.accuracy = None
        #     # self.loss = self.lossfun(outputs[-1], t)
        #     # reporter.report({'loss': self.loss}, self)
        #     # self.accuracy = self.accfun(outputs[-1], t)
        #     # reporter.report({'accuracy': self.accuracy}, self)
        #     return self.loss
        # else:
        #     # return self.model(x)
        #     return outputs[-1]

class CGP2CNN_autoencoder(nn.Module):
    def __init__(self, cgp, in_channel, n_class, imgSize):
        super(CGP2CNN_autoencoder, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []
        self.decode = []
        self.channel_num = [None for _ in range(len(self.cgp))]
        self.size = [None for _ in range(len(self.cgp))]
        self.channel_num[0] = in_channel
        self.size[0] = imgSize
        # encoder
        i = 0
        for name, in1 in self.cgp:
            if name == 'pool_max':
                # self.encode.append(nn.MaxPool2d(self.pool_size, self.pool_size, return_indices=True))
                self.encode.append(MyMaxPooling(self.pool_size, self.channel_num[in1]))
                tmp = self.size[in1]
                tmp = int(tmp / 2)
                self.size[i] = tmp
                tmp = self.channel_num[in1]
                self.channel_num[i] = tmp
            elif name == 'pool_ave':
                self.encode.append(MyAvgPooling(self.pool_size, self.channel_num[in1]))
                tmp = self.size[in1]
                tmp = int(tmp / 2)
                self.size[i] = tmp
                tmp = self.channel_num[in1]
                self.channel_num[i] = tmp
            # elif name == 'concat':
            #     out_size = self.channel_num[in1]+self.channel_num[in2]
            #     self.encode.append(ConvBlock_cat(out_size, out_size, 3))
            #     self.channel_num[i] = out_size
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            # elif name == 'sum':
            #     # channel size check
            #     in_data = [self.channel_num[in1], self.channel_num[in2]]
            #     small_ch_id, large_ch_id = (0, 1) if in_data[0] < in_data[1] else (1, 0)
            #     self.encode.append(ConvBlock_sum(in_data[large_ch_id], in_data[large_ch_id], 3))
            #     self.channel_num[i] = in_data[large_ch_id]
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            elif name == 'ConvBlock1_1':
                out_size = 1
                # self.encode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.channel_num[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif name == 'ConvBlock3_1':
                out_size = 3
                # self.encode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.channel_num[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif name == 'ConvBlock32_3':
                out_size = 32
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock32_5':
                out_size = 32
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock32_7':
                out_size = 32
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_3':
                out_size = 64
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_5':
                out_size = 64
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock64_7':
                out_size = 64
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_3':
                out_size = 128
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_5':
                out_size = 128
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 5))                
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ConvBlock128_7':
                out_size = 128
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 7))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock32_3':
                out_size = 32
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 3))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock32_5':
                out_size = 32
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 5))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock32_7':
                out_size = 32
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 7))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock64_3':
                out_size = 64
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 3))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock64_5':
                out_size = 64
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 5))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock64_7':
                out_size = 64
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 7))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock128_3':
                out_size = 128
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 3))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock128_5':
                out_size = 128
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 5))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock128_7':
                out_size = 128
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 7))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'full':
                input_num = int(self.size[in1] * self.size[in1] * self.channel_num[in1])
                self.encode.append(nn.Linear(input_num, n_class))
            i += 1

        # decoder
        self.channel_num_d = [None for _ in range(len(self.cgp))]
        i -= 2 # 最終層分をとばす
        self.channel_num_d[0] = self.channel_num[i]
        self.channel_num_d[1] = self.channel_num[i]
        i = 0
        self.cgp_inverse = copy.deepcopy(self.cgp)
        self.cgp_inverse.reverse()
        for j in range(len(self.cgp_inverse)):
            self.cgp_inverse[j][1] = int(math.fabs(self.cgp_inverse[j][1]-(len(self.cgp_inverse)-3)))
        for j in range(len(self.cgp_inverse)):
            if j == 0:
                i += 1
                continue
            if self.cgp_inverse[j][0] == 'pool_max':
                # self.decode.append(nn.MaxUnpool2d(self.pool_size, self.pool_size))
                self.decode.append(MyMaxUnPooling(self.pool_size, self.channel_num_d[self.cgp_inverse[j][1]]))
                self.channel_num_d[i] = self.channel_num_d[self.cgp_inverse[j][1]]
            elif self.cgp_inverse[j][0] == 'pool_ave':
                self.decode.append(MyMaxUnPooling(self.pool_size, self.channel_num_d[self.cgp_inverse[j][1]]))
                self.channel_num_d[i] = self.channel_num_d[self.cgp_inverse[j][1]]
            # elif name == 'concat':
            #     out_size = self.channel_num[in1]+self.channel_num[in2]
            #     self.encode.append(ConvBlock_cat(out_size, out_size, 3))
            #     self.channel_num[i] = out_size
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            # elif name == 'sum':
            #     # channel size check
            #     in_data = [self.channel_num[in1], self.channel_num[in2]]
            #     small_ch_id, large_ch_id = (0, 1) if in_data[0] < in_data[1] else (1, 0)
            #     self.encode.append(ConvBlock_sum(in_data[large_ch_id], in_data[large_ch_id], 3))
            #     self.channel_num[i] = in_data[large_ch_id]
            #     tmp = self.size[in1]
            #     self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock1_1':
                out_size = 1
                # self.encode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.channel_num[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock3_1':
                out_size = 3
                # self.encode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.decode.append(ConvBlock(self.channel_num[in1], out_size, 1))
                # self.channel_num[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock32_3':
                out_size = 32
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock32_5':
                out_size = 32
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock32_7':
                out_size = 32
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock64_3':
                out_size = 64
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock64_5':
                out_size = 64
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock64_7':
                out_size = 64
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock128_3':
                out_size = 128
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock128_5':
                out_size = 128
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))                
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ConvBlock128_7':
                out_size = 128
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                self.channel_num_d[i] = out_size
                # tmp = self.size[in1]
                # self.size[i] = tmp
            elif self.cgp_inverse[j][0] == 'ResBlock32_3':
                out_size = 32
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock32_5':
                out_size = 32
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock32_7':
                out_size = 32
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock64_3':
                out_size = 64
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock64_5':
                out_size = 64
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock64_7':
                out_size = 64
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock128_3':
                out_size = 128
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 3))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock128_5':
                out_size = 128
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 5))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock128_7':
                out_size = 128
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full':
                input_num = int(self.size[in1] * self.size[in1] * self.channel_num[in1])
                self.encode.append(nn.Linear(input_num, n_class))
            i += 1
        # 最終層
        for j in range(1):
            if self.cgp_inverse[j][0] == 'ConvBlock1_1':
                out_size = 1
                self.decode.append(ConvBlock_last(self.channel_num_d[i-2], out_size, 1))
                self.channel_num_d[-1] = out_size
            elif self.cgp_inverse[j][0] == 'ConvBlock3_1':
                out_size = 3
                self.decode.append(ConvBlock_last(self.channel_num_d[i-2], out_size, 1))
                self.channel_num_d[-1] = out_size
            else:
                print("error") 
        # self.decode.reverse()
        self.network = self.encode + self.decode
        self.layer_module = nn.ModuleList(self.network)
        # print(self.layer_module)
        # self.model = nn.Sequential(self.arch)
        self.train = True
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp))]
        self.outputs_d = [None for _ in range(len(self.cgp_inverse))]
        self.param_num = 0

    def main(self,x):
        out = x
        outputs = self.outputs
        outputs[0] = x    # input image
        outputs_d = self.outputs_d
        nodeID = 1
        poolID_i = 0
        poolID_o = 0
        _indices = [None for _ in range(len(self.decode))]
        _out_shape = [None for _ in range(len(self.decode))]
        decodeID = 1
        flag = True
        # print('cgp', self.cgp)
        # print('cgp_i', self.cgp_inverse)
        for layer in self.layer_module:
            # encoder
            if nodeID <= len(self.encode):
                # print('encode')
                # print(layer)
                if isinstance(layer, torch.nn.modules.linear.Linear):
                    tmp = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
                    outputs[nodeID] = layer(tmp)
                elif isinstance(layer, MyMaxPooling):
                # elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                    if outputs[self.cgp[nodeID][1]].size(2) > 1:
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                        # poolID_i += 1
                        poolID_o += 1
                    else:
                        outputs[nodeID] = outputs[self.cgp[nodeID][1]]
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        # poolID_i += 1
                        poolID_o += 1
                elif isinstance(layer, MyAvgPooling):
                    if outputs[self.cgp[nodeID][1]].size(2) > 1:
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                        # poolID_i += 1
                        poolID_o += 1
                    else:
                        outputs[nodeID] = outputs[self.cgp[nodeID][1]]
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        # poolID_i += 1
                        poolID_o += 1
                else:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            # decoder
            elif nodeID < (len(self.decode)+len(self.encode)):
                # print('decode')
                # print(layer)
                if flag:
                    outputs_d[0] = outputs[nodeID-1]
                    outputs_d[1] = outputs[nodeID-1]
                    del outputs
                    flag = False
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        print('error')
                        # tmp = outputs_d[self.cgp_inverse[nodeID][1]].view(outputs_d[self.cgp_inverse[nodeID][1]].size(0), -1)
                        # outputs_d[nodeID] = layer(tmp)
                    elif isinstance(layer, MyMaxUnPooling):
                    # elif isinstance(layer, torch.nn.modules.pooling.MaxUnpool2d):
                        # print('pool')
                        # poolID_i -= 1
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                        # outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], _indices[poolID_i])
                    elif isinstance(layer, MyAvgPooling):
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                else:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        print('error')
                        tmp = outputs_d[self.cgp_inverse[decodeID][1]].view(outputs_d[self.cgp_inverse[decodeID][1]].size(0), -1)
                        outputs_d[decodeID] = layer(tmp)
                    elif isinstance(layer, MyMaxUnPooling):
                        # print('pool')
                        # poolID_i -= 1
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                        # outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], _indices[poolID_i])
                    elif isinstance(layer, MyAvgPooling):
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                decodeID += 1
            nodeID += 1
        # 最終層
        layer = self.layer_module[-1]
        # print('decode_f')
        # print(layer)
        # print(outputs_d[decodeID-2].size())
        outputs_d[decodeID] = layer(outputs_d[decodeID-1])
        # print('finish')
        out = outputs_d[-1]
        del outputs_d
        return out
        # return outputs[-1]

    def forward(self, x, t):
        return self.main(x)





class CGP2CNN_autoencoder_full(nn.Module):
    def __init__(self, cgp, in_channel, n_class, imgSize):
        super(CGP2CNN_autoencoder_full, self).__init__()
        self.cgp = cgp
        self.pool_size = 2
        self.arch = OrderedDict()
        self.encode = []
        self.decode = []
        self.channel_num = [None for _ in range(len(self.cgp))]
        self.size = [None for _ in range(len(self.cgp))]
        self.channel_num[0] = in_channel*imgSize*imgSize
        self.size[0] = imgSize
        self.imgSize = imgSize
        self.in_channel = in_channel
        # encoder
        i = 0
        for name, in1 in self.cgp:
            if name == 'pool_max':
                # self.encode.append(nn.MaxPool2d(self.pool_size, self.pool_size, return_indices=True))
                self.encode.append(MyMaxPooling(self.pool_size, self.channel_num[in1]))
                tmp = self.size[in1]
                tmp = int(tmp / 2)
                self.size[i] = tmp
                tmp = self.channel_num[in1]
                self.channel_num[i] = tmp
            elif name == 'ConvBlock32_3':
                out_size = 32
                self.encode.append(ConvBlock(self.channel_num[in1], out_size, 3))
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'ResBlock128_7':
                out_size = 128
                self.encode.append(ResBlock(self.channel_num[in1], out_size, 7))
                if out_size < self.channel_num[in1]:
                    out_size = self.channel_num[in1]
                self.channel_num[i] = out_size
                tmp = self.size[in1]
                self.size[i] = tmp
            elif name == 'full256':
                out_size = 256
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full512':
                out_size = 512
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full1024':
                out_size = 1024
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full2048':
                out_size = 2048
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full4096':
                out_size = 4096
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full8192':
                out_size = 8192
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full16384':
                out_size = 16384
                self.encode.append(nn.Linear(self.channel_num[in1], out_size))
                self.channel_num[i] = out_size
            elif name == 'full':
                out_size = in_channel*imgSize*imgSize
            i += 1

        # decoder
        self.channel_num_d = [None for _ in range(len(self.cgp))]
        i -= 2 # 最終層分をとばす
        self.channel_num_d[0] = self.channel_num[i]
        self.channel_num_d[1] = self.channel_num[i]
        i = 0
        self.cgp_inverse = copy.deepcopy(self.cgp)
        self.cgp_inverse.reverse()
        for j in range(len(self.cgp_inverse)):
            self.cgp_inverse[j][1] = int(math.fabs(self.cgp_inverse[j][1]-(len(self.cgp_inverse)-3)))
        for j in range(len(self.cgp_inverse)):
            if j == 0:
                i += 1
                continue
            if self.cgp_inverse[j][0] == 'pool_max':
                # self.decode.append(nn.MaxUnpool2d(self.pool_size, self.pool_size))
                self.decode.append(MyMaxUnPooling(self.pool_size, self.channel_num_d[self.cgp_inverse[j][1]]))
                self.channel_num_d[i] = self.channel_num_d[self.cgp_inverse[j][1]]
            elif self.cgp_inverse[j][0] == 'ConvBlock128_7':
                out_size = 128
                self.decode.append(ConvBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'ResBlock128_7':
                out_size = 128
                self.decode.append(ResBlock(self.channel_num_d[self.cgp_inverse[j][1]], out_size, 7))
                if out_size < self.channel_num_d[self.cgp_inverse[j][1]]:
                    out_size= self.channel_num_d[self.cgp_inverse[j][1]]
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full256':
                out_size = 256
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full512':
                out_size = 512
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full1024':
                out_size = 1024
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full2048':
                out_size = 2048
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full4096':
                out_size = 4096
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full8192':
                out_size = 8192
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full16384':
                out_size = 16384
                self.decode.append(nn.Linear(self.channel_num_d[self.cgp_inverse[j][1]], out_size))
                self.channel_num_d[i] = out_size
            elif self.cgp_inverse[j][0] == 'full':
                out_size = in_channel*imgSize*imgSize
            i += 1
        # 最終層
        for j in range(1):
            if self.cgp_inverse[j][0] == 'ConvBlock1_1':
                out_size = 1
                self.decode.append(ConvBlock_last(self.channel_num_d[i-2], out_size, 1))
                self.channel_num_d[-1] = out_size
            elif self.cgp_inverse[j][0] == 'ConvBlock3_1':
                out_size = 3
                self.decode.append(ConvBlock_last(self.channel_num_d[i-2], out_size, 1))
                self.channel_num_d[-1] = out_size
            elif self.cgp_inverse[j][0] == 'full':
                out_size = in_channel*imgSize*imgSize
                self.decode.append(nn.Linear(self.channel_num_d[i-2], out_size))
                self.channel_num_d[-1] = out_size
            else:
                print("error") 
        # self.decode.reverse()
        self.network = self.encode + self.decode
        self.layer_module = nn.ModuleList(self.network)
        # print(self.layer_module)
        # self.model = nn.Sequential(self.arch)
        self.train = True
        self.loss = None
        self.accuracy = None
        self.outputs = [None for _ in range(len(self.cgp))]
        self.outputs_d = [None for _ in range(len(self.cgp_inverse))]
        self.param_num = 0

    def main(self,x):
        outputs = self.outputs
        outputs[0] = x.view(x.size(0), -1)
        outputs_d = self.outputs_d
        nodeID = 1
        poolID_i = 0
        poolID_o = 0
        _indices = [None for _ in range(len(self.decode))]
        _out_shape = [None for _ in range(len(self.decode))]
        decodeID = 1
        flag = True
        # print('cgp', self.cgp)
        # print('cgp_i', self.cgp_inverse)
        for layer in self.layer_module:
            # encoder
            if nodeID <= len(self.encode):
                # print('encode')
                # print(layer)
                if isinstance(layer, torch.nn.modules.linear.Linear):
                    # tmp = outputs[self.cgp[nodeID][1]].view(outputs[self.cgp[nodeID][1]].size(0), -1)
                    outputs[nodeID] = F.sigmoid(layer(outputs[self.cgp[nodeID][1]]))
                elif isinstance(layer, MyMaxPooling):
                # elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d) or isinstance(layer, torch.nn.modules.pooling.AvgPool2d):
                    if outputs[self.cgp[nodeID][1]].size(2) > 1:
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                        # poolID_i += 1
                        poolID_o += 1
                    else:
                        outputs[nodeID] = outputs[self.cgp[nodeID][1]]
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        # poolID_i += 1
                        poolID_o += 1
                elif isinstance(layer, MyAvgPooling):
                    if outputs[self.cgp[nodeID][1]].size(2) > 1:
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
                        # poolID_i += 1
                        poolID_o += 1
                    else:
                        outputs[nodeID] = outputs[self.cgp[nodeID][1]]
                        _out_shape[poolID_o] = outputs[self.cgp[nodeID][1]].size()
                        # poolID_i += 1
                        poolID_o += 1
                else:
                    outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            # decoder
            elif nodeID < (len(self.decode)+len(self.encode)):
                # print('decode')
                # print(layer)
                if flag:
                    outputs_d[0] = outputs[nodeID-1]
                    outputs_d[1] = outputs[nodeID-1]
                    del outputs
                    flag = False
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        # tmp = outputs_d[self.cgp_inverse[decodeID][1]].view(outputs_d[self.cgp_inverse[decodeID][1]].size(0), -1)
                        outputs_d[decodeID] = F.sigmoid(layer(outputs_d[self.cgp_inverse[decodeID][1]]))
                    elif isinstance(layer, MyMaxUnPooling):
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                    elif isinstance(layer, MyAvgPooling):
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                else:
                    if isinstance(layer, torch.nn.modules.linear.Linear):
                        # tmp = outputs_d[self.cgp_inverse[decodeID][1]].view(outputs_d[self.cgp_inverse[decodeID][1]].size(0), -1)
                        outputs_d[decodeID] = F.sigmoid(layer(outputs_d[self.cgp_inverse[decodeID][1]]))
                    elif isinstance(layer, MyMaxUnPooling):
                        # print('pool')
                        # poolID_i -= 1
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                        # outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]], _indices[poolID_i])
                    elif isinstance(layer, MyAvgPooling):
                        poolID_o -= 1
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                    else:
                        outputs_d[decodeID] = layer(outputs_d[self.cgp_inverse[decodeID][1]])
                decodeID += 1
            nodeID += 1
        # 最終層
        layer = self.layer_module[-1]
        # print('decode_f')
        # print(layer)
        # print(outputs_d[decodeID-2].size())
        outputs_d[decodeID] = F.sigmoid(layer(outputs_d[decodeID-1]))
        # print('finish')
        out = outputs_d[-1]
        # out = out.view(out.size(0), self.in_channel, self.imgSize, self.imgSize)
        del outputs_d
        return out
        # return outputs[-1]

    def forward(self, x, t):
        return self.main(x)
