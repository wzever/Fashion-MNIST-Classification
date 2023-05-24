import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from dataset import get_data, normalize
from train import *
from eval import *
import argparse
from matplotlib import pyplot as plt
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and evaluate classification net.')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--lr_period', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0, help='GPU index to run')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    ########################################################################
    # 以上加载的数据为 numpy array格式
    # 如果希望使用pytorch或tensorflow等库，需要使用相应的函数将numpy arrray转化为tensor格式
    # 以pytorch为例：
    #   使用torch.from_numpy()函数可以将numpy array转化为pytorch tensor
    #
    # Hint:可以考虑使用torch.utils.data中的class来简化划分mini-batch的操作，比如以下class：
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    ########################################################################

    ########################################################################
    ######################## Implement you code below #######################
    ########################################################################
    args = parse_arguments()
    set_seed(args.seed)
    train_flag, eval_flag = args.train, args.eval

    X_train, X_test, Y_train, Y_test = map(torch.from_numpy, [X_train, X_test, Y_train, Y_test])

    train_set = TensorDataset(X_train, Y_train)
    test_set = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if train_flag:
        train(args, train_loader, test_loader)
    elif eval_flag:
        pretrained = 'pretrained/6-93.7201.pt'
        evaluate(args, test_loader, pretrained)