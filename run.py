# coding: UTF-8
import time
import argparse

import torch
import numpy as np

from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='ERNIE', help='choose a model: Bert, ERNIE')
parser.add_argument('--seed', type=int, default=1108, help='use seed to freeze the result')
args = parser.parse_args()

def seed_freeze(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    if args.seed is not None:
        seed_freeze(args.seed)
    dataset = './Dataset_baidu/'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
