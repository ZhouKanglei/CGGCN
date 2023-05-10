import argparse
import os
import random
import re

import numpy as np
import prettytable as prettytable
import torch
import yaml

yaml.warnings({'YAMLLoadWarning': False})


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'----------------------------------------------------------------\n'
          f'Total params: {total_params:,d} ({total_params / 1000 / 1000:.2f} M)\n'
          f'Trainable params: {train_params:,d} ({train_params / 1000 / 1000:.2f} M)\n'
          f'----------------------------------------------------------------')

def statistic_model_acc(work_dir='./work_dir/ntu/xsub/'):
    os.makedirs(work_dir, exist_ok=True)

    tb = prettytable.PrettyTable()
    tb.field_names = ['Path', 'Model', 'Params', 'Epoch', 'Acc (%)', 'LR']
    tb.align = 'l'
    tb.align['Acc'] = 'r'
    tb.sortby = 'Path'
    tb.reversesort = False

    file_path = os.path.join(work_dir, 'statistics.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                # print(os.path.join(root, file))
                if file == 'config.yaml' or file == 'train.yaml':
                    print(os.path.join(root, file))

                    # load config
                    processor_path = os.path.join(root, '../log/train.log') if \
                        os.path.isfile(os.path.join(root, '../log/train.log')) \
                        else os.path.join(root, '../processor.log')
                    if os.path.isfile(processor_path):
                        with open(processor_path, 'r', encoding='utf-8') as f_processor:
                            log = f_processor.read()
                            f_processor.close()

                            result = re.search('Model total number of params:.*\)', log)
                            model_param = result.group()[30:] if result else ''

                    else:
                        model_param = ''

                    # load config
                    config_path = os.path.join(root, 'train.yaml')
                    if os.path.isfile(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f_config:
                            default_arg = yaml.load(f_config)
                            model_name = default_arg['model']
                            f_config.close()

                    else:
                        model_name = ''

                    # best acc
                    acc_path = os.path.join(root, '../best/acc.txt')
                    if os.path.exists(acc_path):
                        with open(acc_path, 'r', encoding='utf-8') as f_acc:
                            acc = f'{float(f_acc.read()) * 100:.2f}'

                            f_acc.close()
                    else:
                        acc = ''

                    # best epoch
                    best_path = os.path.join(root, '../best')
                    if os.path.isdir(best_path):
                        files = os.listdir(best_path)
                        for model_weight in files:
                            if '.pt' in model_weight:
                                epoch = model_weight.split('-')[1]
                                break
                            else:
                                epoch = ''
                    else:
                        epoch = ''

                    # load config
                    processor_path = os.path.join(root, '../log/train.log') if \
                        os.path.isfile(os.path.join(root, '../log/train.log')) \
                        else os.path.join(root, '../processor.log')
                    if os.path.isfile(processor_path):
                        with open(processor_path, 'r', encoding='utf-8') as f_processor:
                            log = f_processor.read()
                            f_processor.close()

                            result = re.search(f'Training epoch: {epoch}, LR: .*\n', log)
                            lr = result.group().split(':')[-1].strip() if result else ''

                    else:
                        lr = ''

                    tb.add_row([root[11:], model_name, model_param, epoch, acc, lr])

        f.write(tb.get_string())

        print(tb)


if __name__ == '__main__':
    statistic_model_acc(work_dir='./work_dir/ntu/xsub/')