#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd

from cgp import *
from cgp_config import *
from cnn_train import CNN_train


if __name__ == '__main__':

    func_set = {
        'ConvSet': CgpInfoConvSet,
        'ResSet': CgpInfoResSet,
        'Primitive': CgpInfoPrimitiveSet,
    }

    parser = argparse.ArgumentParser(description='Evolving CNN structures of GECCO 2017 paper')
    parser.add_argument('--func_set', '-f', choices=func_set.keys(), default='ConvSet', help='Function set of CGP (ConvSet or ResSet)')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--log_file', default='./log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = func_set[args.func_set](rows=1, cols=30, level_back=5, min_active_num=5, max_active_num=30)
        with open(args.net_info_file, mode='wb') as f:
            pickle.dump(network_info, f)
        # Evaluation function for CGP (training CNN and return validation accuracy)
        imgSize = 64
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset='bsds', valid_data_ratio=0.1, verbose=True, epoch_num=30, batchsize=16, imgSize=imgSize)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init)
        cgp.modified_evolution(max_eval=10000, mutation_rate=0.1, log_file=args.log_file)

    # --- Retraining evolved architecture ---
    elif args.mode == 'retrain':
        print('Retrain')
        # # In the case of existing log_cgp.txt
        # # Load CGP configuration
        with open(args.net_info_file, mode='rb') as f:
            network_info = pickle.load(f)

        # Load network architecture
        cgp = CGP(network_info, None)
        data = pd.read_csv(args.log_file, header=None)  # Load log file
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))  # Read the log at final generation
        print(cgp._log_data(net_info_type='active_only', start_time=0))

        # Retraining the network
        temp = CNN_train('bsds', validation=False, verbose=True)
        acc = temp(cgp.pop[0].active_net_list(), 0, epoch_num=500, batchsize=16, weight_decay=1e-4, eval_epoch_num=450,
                   data_aug=True, comp_graph=None, out_model='retrained_net.model', init_model=None)
        print(acc)

        # # otherwise
        # temp = CNN_train('bsds', validation=False, verbose=True)
        # cgp = [['input', 0], ['ConvBlock64_5', 0], ['ConvBlock128_3', 1], ['ConvBlock32_3', 2], ['ConvBlock32_3', 3], ['ConvBlock128_3', 4], ['pool_max', 5], ['ConvBlock1_1', 6]]
        # acc = temp(cgp, 1, epoch_num=1000, batchsize=10, weight_decay=1e-4, eval_epoch_num=450,
        #            data_aug=True, comp_graph=None, out_model='retrained_net.model', init_model=None)
        # print(acc)

    elif args.mode == 'reevolution':
        # # restart
        print('Restart!!')
        imgSize = 160
        with open('network_info.pickle', mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset='bsds', valid_data_ratio=0.1, verbose=True, epoch_num=10, batchsize=10, imgSize=imgSize)
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize)

        data = pd.read_csv('./log_cgp.txt', header=None)
        cgp.load_log(list(data.tail(1).values.flatten().astype(int)))
        cgp.modified_evolution(max_eval=2000, mutation_rate=0.1, log_file='./log_restat.txt')

    else:
        print('Undefined mode.')
