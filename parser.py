# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021/8/13 9:19
import argparse
import os
import yaml

from utils import str2bool

class cmdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + '_non_default', True)

class Parser(object):
    """parameter priority: command line > config file > default"""

    def __init__(self, config_file):
        self.init_parser(config_file)
        self.merge_config()
        self.check_args()

    def init_parser(self, config_file):
        parser = argparse.ArgumentParser(description='EGGCN')

        parser.add_argument('--work_dir', type=str, default='./work_dir/temp',
                            action=cmdAction, help='the work folder for storing results')
        parser.add_argument('--model_saved_name', default='')
        parser.add_argument('--config', default=config_file,
                            action=cmdAction, help='path to the configuration file')
        parser.add_argument('--assume_yes', default=True,
                            action=cmdAction, help='Say yes to every prompt')

        parser.add_argument('--phase', default='train', action=cmdAction, help='must be train or test')
        parser.add_argument('--save_score', type=str2bool, default=True,
                            action=cmdAction, help='if true, the classification score will be stored')

        parser.add_argument('--seed', type=int, default=1024, action=cmdAction, help='random seed')
        parser.add_argument('--log_interval', type=int, default=100,
                            action=cmdAction, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=1, action=cmdAction, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=1,
                            action=cmdAction, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--eval_start', type=int, default=1, action=cmdAction, help='The epoch number to start evaluating models')
        parser.add_argument('--print_log', type=str2bool, default=True, action=cmdAction, help='print logging or not')
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            action=cmdAction, help='which Top K accuracy will be shown')

        parser.add_argument('--feeder', default='feeder.feeder', action=cmdAction, help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=16,
                            action=cmdAction, help='the number of worker for data loader')
        parser.add_argument('--window_size', type=int, default=64, action=cmdAction, help='window size of dataloader')
        parser.add_argument('--train_feeder_args', default=dict(), action=cmdAction, help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', default=dict(), action=cmdAction, help='the arguments of data loader for test')

        parser.add_argument('--model', default=None, action=cmdAction, help='the model will be used')
        parser.add_argument('--model_args', type=dict, default=dict(), action=cmdAction, help='the arguments of model')
        parser.add_argument('--weights', default=None, action=cmdAction, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            action=cmdAction, help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--continue_train', default=True, action=cmdAction, help='whether load the best pre-trained model')
        parser.add_argument('--half', default=False, action=cmdAction, help='Use half-precision (FP16) training')
        parser.add_argument('--amp_opt_level', type=int, default=1, action=cmdAction, help='NVIDIA Apex AMP optimization level')

        parser.add_argument('--base_lr', type=float, default=0.01, action=cmdAction, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+',
                            action=cmdAction, help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            action=cmdAction, help='the indexes of GPUs for training or testing')
        parser.add_argument('--optimizer', default='SGD', action=cmdAction, help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=False, action=cmdAction, help='use nesterov or not')
        parser.add_argument('--batch_size', type=int, default=32, action=cmdAction, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, action=cmdAction, help='test batch size')
        parser.add_argument('--forward_batch_size', type=int, default=16,
                            action=cmdAction, help='Batch size during forward pass, must be factor of --batch-size')
        parser.add_argument('--start_epoch', type=int, default=0, action=cmdAction, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, action=cmdAction, help='stop training in which epoch')
        parser.add_argument('--weight_decay', type=float, default=0.0005, action=cmdAction, help='weight decay for optimizer')
        parser.add_argument('--optimizer_states', type=str, action=cmdAction, help='path of previously saved optimizer states')
        parser.add_argument('--checkpoint', type=str, action=cmdAction, help='path of previously saved training checkpoint')
        parser.add_argument('--debug', type=str2bool, default=False, action=cmdAction, help='Debug mode; default false')
        parser.add_argument('--data_proportion', type=float, default=1, action=cmdAction, help='Use x% data for training')
        parser.add_argument('--fine_tune', type=str2bool, default=False, action=cmdAction, help='Fine tune the pre-trained ensemble model')
        parser.add_argument('--bone', type=str2bool, default=False, action=cmdAction, help='bone modality')
        parser.add_argument('--vel', type=str2bool, default=False, action=cmdAction, help='vel modality')
        parser.add_argument('--multi_input', type=str2bool, default=False, action=cmdAction, help='if multi-input modalities are used')

        parser.add_argument('--ens_model_weights', type=list, default=[], action=cmdAction, help='Pre-trained ensemble model weights')
        parser.add_argument('--optimizer_args', type=dict, default=dict(), action=cmdAction, help='the arguments of the optimizer')

        self.parser = parser

    def get_args(self):
        return  self.args

    def merge_config(self):
        self.args = self.parser.parse_args()

        if self.args.config is not None:
            with open(self.args.config, 'r') as f:
                default_arg = yaml.load(f)

        for k, v in default_arg.items():
            if k not in vars(self.args).keys():
                setattr(self.args, k, v)

            elif not hasattr(self.args, f'{k}_non_default'):
                setattr(self.args, k, v)
                # print(f'set {k} to {v}')

    def check_args(self):
        # work dir
        if hasattr(self.args.model_args, 'num_groups'):
            num_groups_str = '-' + str(self.args.model_args['num_groups'][0]) + '_' + \
                             str(self.args.model_args['num_groups'][1]) + '_' + \
                             str(self.args.model_args['num_groups'][2])
            if num_groups_str not in self.args.work_dir:
                self.args.work_dir += num_groups_str

            num_gcn_str = '-gcn_' + str(self.args.model_args['num_gcn_scales'])
            if num_gcn_str not in self.args.work_dir:
                self.args.work_dir += num_gcn_str

        if hasattr(self.args, 'bone_non_default'):
            self.args.train_feeder_args['bone'] = self.args.bone
            self.args.test_feeder_args['bone'] = self.args.bone
        if 'bone' not in self.args.train_feeder_args.keys():
            self.args.train_feeder_args['bone'] = False
        if 'bone' not in self.args.test_feeder_args.keys():
            self.args.test_feeder_args['bone'] = False

        if hasattr(self.args, 'vel_non_default'):
            self.args.train_feeder_args['vel'] = self.args.vel
            self.args.test_feeder_args['vel'] = self.args.vel
        if 'vel' not in self.args.train_feeder_args.keys():
            self.args.train_feeder_args['vel'] = False
        if 'vel' not in self.args.test_feeder_args.keys():
            self.args.test_feeder_args['vel'] = False

        if hasattr(self.args, 'multi_input_non_default'):
            self.args.train_feeder_args['multi_input'] = self.args.multi_input
        if 'multi_input' not in self.args.train_feeder_args.keys():
            if not self.args.train_feeder_args['vel']:
                self.args.train_feeder_args['multi_input'] = False
            else:
                self.args.train_feeder_args['multi_input'] = True

        if self.args.phase == 'train':
            if self.args.train_feeder_args['bone'] == True and '-bone' not in self.args.work_dir:
                self.args.work_dir += '-bone'

            if self.args.train_feeder_args['vel'] == True and '-vel' not in self.args.work_dir:
                self.args.work_dir += '-vel'

            self.args.test_feeder_args['bone'] = self.args.train_feeder_args['bone']
            self.args.test_feeder_args['vel'] = self.args.train_feeder_args['vel']
            self.args.test_feeder_args['multi_input'] = self.args.train_feeder_args['multi_input']

            if hasattr(self.args, 'window_size_non_default'):
                self.args.train_feeder_args['window_size'] = self.args.window_size
        else:
            if self.args.test_feeder_args['bone'] == True and '-bone' not in self.args.work_dir:
                self.args.work_dir += '-bone'

            if self.args.test_feeder_args['vel'] == True and '-vel' not in self.args.work_dir:
                self.args.work_dir += '-vel'



    def form_config(self):
        # load arg from config file
        p = self.parser.parse_args()
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    assert (k in key), 'WRONG ARG: %s' % k

            self.parser.set_defaults(**default_arg)
