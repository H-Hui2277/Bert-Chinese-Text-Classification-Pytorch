# coding: UTF-8
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn import metrics

from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    # log info
    with open(config.log_file, mode='a+', encoding='utf-8') as f:
        for k, v in config.__dict__.items():
            f.write(f'{k}: {v}\n')
        f.close()

    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0.
    test_best_acc = 0.
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            # balanced weights
            class_counts = torch.bincount(labels, minlength=len(config.class_list)).float().to(config.device)
            total_samples = class_counts.sum()
            class_weights = total_samples / (len(config.class_list) * class_counts)

            loss = F.cross_entropy(outputs, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            if total_batch % 1000 == 0 and total_batch != 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                improve = ''
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), os.path.join(config.save_path, 'best_val.pt'))
                    improve += '*'
                test_acc, test_loss = evaluate(config, model, test_iter)
                if test_acc > test_best_acc:
                    test_best_acc = test_acc
                    torch.save(model.state_dict(), os.path.join(config.save_path, 'best_test.pt'))
                    improve += '^'

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Test Loss: {5:>5.2}, Test Acc: {6:>6.2%}  Time: {7} {8}'
                msg = msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, test_loss, test_acc, time_dif, improve)
                # Log
                with open(config.log_file, mode='a+') as f:
                    f.write(f'{msg}\n')
                    f.close()
                print(msg)
                model.train()
            total_batch += 1
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    # model.load_state_dict(torch.load(os.path.join(config.save_path, 'best_test.pt')))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_f1_score, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("F1-Score...")
    print(test_f1_score)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, labels=config.class_list, digits=4)
        f1_score = metrics.f1_score(labels_all, predict_all, average=None)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), f1_score, confusion
    return acc, loss_total / len(data_iter)
