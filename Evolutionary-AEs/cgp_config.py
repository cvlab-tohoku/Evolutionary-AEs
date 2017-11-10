#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import cnn_train as cnn


# wrapper function for multiprocessing
def arg_wrapper_mp(args):
    return args[0](*args[1:])

# # wrapper function for multiprocessing
# def arg_wrapper_mp(args):
#     return cnn_eval(*args)


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# Evaluation of CNNs
def cnn_eval(net, gpu_id, epoch_num, batchsize, dataset, valid_data_ratio, verbose, imgSize):

    print('\tgpu_id:', gpu_id, ',', net)
    train = cnn.CNN_train(dataset, validation=True, valid_data_ratio=valid_data_ratio, verbose=verbose, imgSize=imgSize)
    evaluation = train(net, gpu_id, epoch_num=epoch_num, batchsize=batchsize,
                               comp_graph='CNN%d.dot'%(gpu_id), out_model=None, init_model=None)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, epoch_num=50, batchsize=64, dataset='cifar10', valid_data_ratio=0.1, verbose=True, imgSize=32):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.valid_data_ratio = valid_data_ratio
        self.verbose = verbose
        self.imgSize = imgSize

    def __call__(self, net_lists):
        evaluations = np.zeros(len(net_lists))
        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i
            # pool = mp.Pool(process_num)
            pool = NoDaemonProcessPool(process_num)
            arg_data = [(cnn_eval, net_lists[i+j], j, self.epoch_num, self.batchsize, self.dataset,
                         self.valid_data_ratio, self.verbose, self.imgSize) for j in range(process_num)]
            # arg_data = [(net_lists[i+j], j, self.epoch_num, self.batchsize, self.dataset,
            #              self.valid_data_ratio, self.verbose) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            # evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations

# class CNNEvaluation(object):
#     def __init__(self, gpu_num, epoch_num=50, batchsize=256, dataset='cifar10', valid_data_ratio=0.1, verbose=True):
#         self.gpu_num = gpu_num
#         self.epoch_num = epoch_num
#         self.batchsize = batchsize
#         self.dataset = dataset
#         self.valid_data_ratio = valid_data_ratio
#         self.verbose = verbose

#     def __call__(self, net_lists):
#         evaluations = np.zeros(len(net_lists))
#         for i in np.arange(0, len(net_lists), self.gpu_num):
#             process_num = np.min((i + self.gpu_num, len(net_lists))) - i
#             pool = mp.Pool(process_num)
#             arg_data = [(cnn_eval, net_lists[i+j], j, self.epoch_num, self.batchsize, self.dataset,
#                          self.valid_data_ratio, self.verbose) for j in range(process_num)]
#             evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
#             pool.terminate()

#         return evaluations


class CgpInfoConvSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['ConvBlock32_3', 'ConvBlock32_5',
                          'ConvBlock64_3', 'ConvBlock64_5',
                          'ConvBlock128_3', 'ConvBlock128_5',
                          'pool_max', 'pool_ave']
                          # 'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1]
                            # 2, 2]

        self.out_num = 1
        self.out_type = ['ConvBlock1_1']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])


class CgpInfoResSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['ResBlock32_3', 'ResBlock32_5',
                          'ResBlock64_3', 'ResBlock64_5',
                          'ResBlock128_3', 'ResBlock128_5',
                          'pool_max', 'pool_ave']
                          # 'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1, 1]
                            # 2, 2]

        self.out_num = 1
        self.out_type = ['ConvBlock1_1']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])

class CgpInfoFullSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        # network configurations depending on the problem
        self.input_num = 1

        self.func_type = ['full256', 'full512',
                          'full1024', 'full2048',
                          'full4096', 'full8192',
                          'full16384']
                          # 'concat', 'sum']
        self.func_in_num = [1, 1,
                            1, 1,
                            1, 1,
                            1]
                            # 2, 2]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])