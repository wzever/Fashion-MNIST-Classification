import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from model import *
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def train(args, train_iter, test_iter):
    # load training dataset 
    batch_size = args.batch_size

    # set hyper-parameters
    lr = args.lr
    wd = args.wd
    lr_decay = args.lr_decay
    num_epochs = args.num_epochs

    # get current device
    device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")

    # get training net
    net = CNN().to(device)
    # print(net)
    # set optimizer & scheduler
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_period, gamma=args.lr_decay)

    # set loss function
    loss_func = nn.CrossEntropyLoss()

    step_n = 0
    best_acc = 0
    train_accs, test_accs, train_loss, test_loss = [], [], [], []

    feat_to_draw = {'conv': None, 'fc': None, 'final': None}
    class_cnt = [0 for _ in range(10)]
    feat_label = []
    imgs_to_draw = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_corr, test_corr = 0, 0
        tt_loss_train, tt_loss_test = 0, 0 
        for i, (img, labels) in enumerate(train_iter): 
            net.train()
            # cur_batch_size = img.shape[0]
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            loss = loss_func(pred, labels.long()).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            _, pred = torch.max(pred.data, dim=1)
            train_corr += pred.eq(labels.data).cpu().sum().item()
            tt_loss_train += loss.cpu().item()
            step_n += 1

        end_time = time.time()
        train_acc = 100.0 * train_corr / batch_size / len(train_iter)
        print(f'epoch {epoch}, train time = {(end_time - start_time):.2f}')

        scheduler.step()
        end_time = time.time()
        train_loss.append(tt_loss_train / len(train_iter))
        train_accs.append(train_acc)

        with torch.no_grad():
            for _, (img, labels) in enumerate(test_iter): 
                net.eval() # test
                img, labels = img.to(device), labels.to(device)

                pred = net(img)
                loss = loss_func(pred, labels.long()).mean()

                _, pred = torch.max(pred.data, dim=1)
                batch_corr = pred.eq(labels.data).cpu().sum().item()
                test_corr += batch_corr
                tt_loss_test += loss.cpu().item()

                feat_map = {'conv': net.conv_feat, 'fc': net.fc_feat, 'final': net.final_feat}

                if epoch == num_epochs - 1:
                    cur_batch_size = img.shape[0]
                    for i in range(cur_batch_size):
                        if class_cnt[int(labels[i])] > 50:
                            continue
                        feat_label.append(int(labels[i]))
                        imgs_to_draw.append(img[i].cpu().numpy())
                        class_cnt[int(labels[i])] += 1
                        for name in feat_map.keys():
                            ori_feat = feat_to_draw[name]
                            cur_feat = feat_map[name][i]
                            if ori_feat is not None:
                                feat_to_draw[name] = torch.cat([ori_feat, cur_feat.view(1, -1)], dim=0)
                            else:
                                feat_to_draw[name] = cur_feat.view(1, -1)

        test_acc = 100.0 * test_corr / batch_size / len(test_iter)

        if not os.path.exists("pretrained"):
            os.mkdir("pretrained")

        if test_acc > best_acc:
            best_acc = test_acc
            if best_acc > 0.92:
                torch.save(net.state_dict(), f"pretrained/{args.seed}-{test_acc:.4f}.pt")

        print(f'train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f} (best: {best_acc:.4f})')
        test_accs.append(test_acc)
        test_loss.append(tt_loss_test / len(test_iter))
        print(f'train_loss = {(train_loss[-1]):.4f}, test_loss = {(test_loss[-1]):.4f}', end='\n\n') 

    feat_to_draw = {name: feat.numpy() for name, feat in feat_to_draw.items()}
    np.save('feat.npy', feat_to_draw)
    np.save('labels.npy', np.array(feat_label))
    np.save('imgs.npy', np.array(imgs_to_draw))
    np.save('train_loss.npy', np.array(train_loss))
    np.save('test_loss.npy', np.array(test_loss))
    np.save('train_accs.npy', np.array(train_accs))
    np.save('test_accs.npy', np.array(test_accs))