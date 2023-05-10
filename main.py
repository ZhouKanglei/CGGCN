# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
yaml.warnings({'YAMLLoadWarning': False})

from parser import Parser
from processor import Processor
from utils import init_seed

if __name__ == '__main__':
    default_config_file = 'config/nturgbd-cross-subject/test23.yaml'
    parser = Parser(default_config_file)
    args = parser.get_args()
    init_seed(args.seed)

    processor = Processor(args)
    processor.start()