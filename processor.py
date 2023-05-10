# !/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021/8/13 7:59

import inspect
import logging
import os
import pathlib
import pickle
import pprint
import shutil
import time
from collections import OrderedDict, defaultdict

import apex as apex
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import import_class, count_params, init_seed


class Processor(object):
    """Processor for Skeleton-based Action Recognition"""

    def __init__(self, args):
        self.args = args

        self.save_args()
        self.init_print_log()
        self.create_tb_log()

        self.global_step = 0
        self.lr = self.args.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        if self.args.half:
            self.logger.info('*************************************')
            self.logger.info('|   Using Half Precision Training   |')
            self.logger.info('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.args.amp_opt_level}'
            )
            if self.args.amp_opt_level != 1:
                self.logger.info('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.logger.info(f'{len(self.args.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=self.output_device
                )

    def load_model(self):
        self.output_device = self.args.device[0] if type(self.args.device) is list else self.args.device
        Model = import_class(self.args.model)

        self.model = Model(**self.args.model_args).cuda(self.output_device)
        self.loss = nn.CrossEntropyLoss().cuda(self.output_device)
        self.logger.info(f'Model total number of params: {count_params(self.model):,d} (%.2f M)' % (
                    count_params(self.model) / 1000 / 1000))

        # find pre-trained model
        if (self.args.continue_train or self.args.phase == 'test') and self.args.weights == None:
            files = pathlib.Path(os.path.join(self.args.work_dir, 'best'))
            # acc
            best_accs = files.glob('acc*.txt')
            for acc in best_accs:
                acc_file = os.path.join(os.path.join(self.args.work_dir, 'best'), acc.name)
                with open(acc_file, 'r', encoding='utf-8') as f:
                    self.best_acc = float(f.read())
                    self.logger.info(f'Loading best accuracy: {self.best_acc * 100:.2f}%')

            # weights
            best_models = files.glob('weights*.pt')
            for model in best_models:
                self.args.weights = os.path.join(os.path.join(self.args.work_dir, 'best'), model.name)
                if self.args.start_epoch == 0 and self.args.continue_train:
                    self.args.start_epoch = int(model.name.split('-')[1])
                self.best_acc_epoch = int(model.name.split('-')[1])

            # checkpoint
            files = pathlib.Path(os.path.join(self.args.work_dir, 'checkpoints'))
            best_models = files.glob(f'checkpoint-{self.args.start_epoch}*.pt')
            for model in best_models:
                self.args.checkpoint = os.path.join(os.path.join(self.args.work_dir, 'checkpoints'), model.name)

        # load pre-trained weights
        if (self.args.continue_train or self.args.phase == 'test') and self.args.weights:
            self.logger.info(f'Loading weights from: {self.args.weights}')

            try:
                self.global_step = int(self.args.weights[:-3].split('-')[-1])
            except:
                self.logger.exception(f'Cannot parse global_step from model weights filename [{self.args.weights}]')
                self.global_step = 0

            if '.pkl' in self.args.weights:
                with open(self.args.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.args.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(self.output_device)] for k, v in weights.items()])

            for w in self.args.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.logger.info(f'Successfully Remove Weights: {w}')
                else:
                    self.logger.info(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.logger.info('Can not find these weights:')
                for d in diff:
                    self.logger.info('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        # copy model file and processor file
        if self.args.phase == 'train':
            src_dir = os.path.join(self.args.work_dir, 'src')
            os.makedirs(src_dir, exist_ok=True)
            shutil.copy2(inspect.getfile(Model), src_dir)
            shutil.copy2(os.path.join('.', __file__), src_dir)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) ofc different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())

        if self.args.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                params,
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f'Unsupported optimizer: {self.args.optimizer}')

        # load optimizer states if any
        if self.args.checkpoint is not None:
            self.logger.info(f'Loading optimizer states from: {self.args.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.args.checkpoint)['optimizer_states'])
            if self.optimizer.param_groups[0]['lr'] < 0.0010:
                self.optimizer.param_groups[0]['lr'] = 0.0010
            self.logger.info(f'Starting LR: {self.optimizer.param_groups[0]["lr"]:.4f}')
            self.logger.info(f'Starting WD: {self.optimizer.param_groups[0]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.step, gamma=0.1)

        if self.args.checkpoint is not None:
            scheduler_states = torch.load(self.args.checkpoint)['lr_scheduler_states']

            self.logger.info(f'Loading LR scheduler states from: {self.args.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.logger.info(f'Starting last epoch: {scheduler_states["last_epoch"] + 1:d}')
            self.logger.info(f'Loading milestones: {scheduler_states["last_epoch"] + 1:d}')

    def load_data(self):
        Feeder = import_class(self.args.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.args.seed + worker_id + 1)

        if self.args.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.args.train_feeder_args),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn
            )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args.test_feeder_args),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn
        )

    def create_tb_log(self):
        """Tensor-board visualization log"""

        if self.args.phase == 'train':
            # Add control through command line
            debug = self.args.train_feeder_args['debug'] or self.args.debug
            log_dir = os.path.join(self.args.work_dir, 'trainlogs')
            if not debug:
                if os.path.isdir(log_dir):
                    self.logger.info(f'Log dir [{log_dir}] already exists')
                    answer = 'y' if self.args.assume_yes else 'n'

                    if answer.lower() in ('y', ''):
                        shutil.rmtree(log_dir)
                        self.logger.info(f'Dir [{log_dir}] removed')
                    else:
                        self.logger.info(f'Dir [{log_dir}] not removed')

                self.train_writer = SummaryWriter(os.path.join(log_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(log_dir, 'val'), 'val')
            else:
                self.debug_writer = SummaryWriter(os.path.join(log_dir, 'debug'), 'debug')

    def init_print_log(self):
        # Logger: CRITICAL > ERROR > WARNING > INFO > DEBUG
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # stream handler
        log_sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_sh.setFormatter(formatter)

        logger.addHandler(log_sh)

        # file handler
        log_dir = os.path.join(self.args.work_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{self.args.phase}.log')
        log_fh = logging.FileHandler(log_file, mode='a')
        log_fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        log_fh.setFormatter(formatter)

        logger.addHandler(log_fh)

        self.logger = logger

    def save_args(self):
         # save args
        args_dict = vars(self.args)

        if self.args.continue_train == False and self.args.phase == 'train':
            shutil.rmtree(self.args.work_dir, ignore_errors=True)
            print(f'----------------------------------------------------------------\n'
                  f'Empty the work dir [{self.args.work_dir}]\n'
                  f'----------------------------------------------------------------')

        os.makedirs(os.path.join(self.args.work_dir, 'config'), exist_ok=True)
        with open(os.path.join(self.args.work_dir, f'config/{self.args.phase}.yaml'), 'w') as f:
            yaml.dump(args_dict, f)

    def save_states(self, states, out_folder, out_name):
        out_folder_path = os.path.join(self.args.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict()
        }

        checkpoint_name = f'checkpoint-{epoch}-{int(self.global_step)}.pt'
        self.save_states(state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('moudle.')[-1], v.cpu()] for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
        self.save_states(weights, out_folder, weights_name)

    def save_best(self, epoch, out_folder='best'):
        out_folder_path = os.path.join(self.args.work_dir, out_folder)
        shutil.rmtree(out_folder_path, ignore_errors=True)
        os.makedirs(out_folder_path, exist_ok=True)

        self.logger.info(f'**** Current best epoch {epoch}, acc: {self.best_acc * 100:.2f}% ****')
        with open(os.path.join(out_folder_path, 'acc.txt'), 'w', encoding='utf-8') as f:
            f.write(f'{self.best_acc}')

        self.save_weights(epoch, out_folder=out_folder)
        self.save_checkpoint(epoch, out_folder=out_folder)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        loader = self.data_loader['train']
        loss_values, acc_values = [], []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f'----------------------------------------------------------------')
        self.logger.info(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label, index) in enumerate(process):
            if np.random.random() > self.args.data_proportion:
                continue

            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            # ------------- Gradient Accumulation for Smaller Batches ------------
            real_batch_size = self.args.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, 'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_label = data[left:right], label[left:right]

                # forward
                outputs = self.model(batch_data)
                l1 = 0
                if isinstance(outputs, tuple):
                    output_1, output_2, output = outputs
                    loss = self.loss(output_1, batch_label) + self.loss(output_2, batch_label)
                    loss = loss / splits / 2
                else:
                    output = outputs
                    loss = self.loss(output, batch_label) / splits

                if self.args.half:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # display loss
                process.set_description(f'(BS {real_batch_size}) loss - {loss.item():.4f}/{np.mean(loss_values):.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())
                acc_values.append(acc.item())

                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.item() * splits, self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            # -------------------------------------------------------------------
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            del output, loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        num_splits = self.args.batch_size // self.args.forward_batch_size
        self.logger.info(
            f'\tMean training loss: {mean_loss:.4f} '
            f'(BS {self.args.batch_size}: {mean_loss * num_splits:.4f})'
            f' Mean training acc: {np.mean(acc_values) * 100:.2f}%')
        self.logger.info('\tTime consumption: [Data] {dataloader}, [Network] {model}'.format(**proportion))

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.args.eval_start:
            return
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.logger.info(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                loss_values = []
                score_batches, score_batches_1, score_batches_2 = [], [], []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    # forward
                    outputs = self.model(data)
                    l1 = 0
                    if isinstance(outputs, tuple):
                        output_1, output_2, output = outputs
                        loss = self.loss(output_1, label) + self.loss(output_2, label)
                        loss /= 2
                    else:
                        output_1 = output_2 = output = outputs
                        loss = self.loss(output, label)

                    score_batches.append(output.data.cpu().numpy())
                    score_batches_1.append(output_1.data.cpu().numpy())
                    score_batches_2.append(output_2.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ', ' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ', ' + str(x) + ', ' + str(true[i]) + '\n')

            score, score_1, score_2 = np.concatenate(score_batches), np.concatenate(score_batches_1), \
                                      np.concatenate(score_batches_2)
            loss = np.mean(loss_values)
            accurary = self.data_loader[ln].dataset.top_k(score, 1)

            if accurary > self.best_acc and self.args.phase == 'train':
                self.best_acc, self.best_acc_epoch = accurary, epoch + 1
                self.save_best(epoch + 1)

            self.logger.info(f'Accuracy: {accurary:.4f} model: {self.args.work_dir}')
            if self.args.phase == 'train' and not self.args.debug:
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accurary, self.global_step)

            # grid search global optimal model
            lambda_1, lambda_2 = 1, 1
            if self.args.phase == 'test':
                # lambda_1, lambda_2 = self.grid_search(score_1, score_2)
                print()

            score = lambda_1 * score_1 + lambda_2 * score_2
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.logger.info(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values):.4f}')
            for k in self.args.show_topk:
                self.logger.info(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            # save_score
            if save_score:
                os.makedirs(os.path.join(self.args.work_dir, 'score'), exist_ok=True)
                with open('{}/epoch{}_{}_score.pkl'.format(os.path.join(self.args.work_dir, 'score'), epoch + 1, ln),
                          'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

        return accurary

    def start(self):
        if self.args.phase == 'train':
            self.logger.info(f'Parameters:\n{pprint.pformat(vars(self.args))}\n')
            self.logger.info(f'Model total number of params: {count_params(self.model):,d} (%.2f M)' % (
                        count_params(self.model) / 1e6))
            if self.global_step == 0:
                self.global_step = self.args.start_epoch * len(self.data_loader['train'])
            for epoch in range(self.args.start_epoch, self.args.num_epoch):

                save_model = ((epoch + 1) % self.args.save_interval == 0) or (epoch + 1 == self.args.num_epoch)
                # Train
                self.train(epoch, save_model=save_model)
                # Evaluation
                acc = self.eval(epoch, save_score=self.args.save_score, loader_name=['test'])
                if acc == self.best_acc and hasattr(self.args, 'num_epoch_non_default'): break
                # Update LR scheduler
                self.lr_scheduler.step()

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.logger.info(f'Best accuracy: {self.best_acc:.4f}')
            self.logger.info(f'Epoch number: {self.best_acc_epoch}')
            self.logger.info(f'Model name: {self.args.work_dir}')
            self.logger.info(f'Model total number of params: {num_params:,d} (%.2f M)'
                             % (count_params(self.model) / 1e6))
            self.logger.info(f'Weight decay: {self.args.weight_decay}')
            self.logger.info(f'Base LR: {self.args.base_lr:.4f}')
            self.logger.info(f'Batch Size: {self.args.batch_size}')
            self.logger.info(f'Forward Batch Size: {self.args.forward_batch_size}')
            self.logger.info(f'Test Batch Size: {self.args.test_batch_size}')

        elif self.args.phase == 'test':
            if not self.args.test_feeder_args['debug']:
                wf = os.path.join(self.args.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.args.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')

            self.logger.info(f'Model: {self.args.model}')
            self.logger.info(f'Weights: {self.args.weights}')

            self.eval(
                epoch=self.best_acc_epoch,
                save_score=self.args.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.logger.info('Done.\n')
