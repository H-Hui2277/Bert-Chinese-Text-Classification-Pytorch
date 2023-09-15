# coding: UTF-8
import time
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.save_path = dataset + f'/saved_dict_{time.strftime("%m%d-%H%M%S")}/'  # 模型训练结果
        self.log_file = os.path.join(self.save_path, 'log.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.val_steps = 1000 # 每隔val_steps在开发集和测试集上验证一次
        
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer:BertTokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_layer_dropout = 0.2
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, output_attentions=True)
        self.dropout = nn.Dropout(config.hidden_layer_dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out


PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号

    
class BertDataset(Dataset):
    def __init__(self, config:Config, mode='train'):
        super().__init__()
        self.config = config
        self.device = config.device
        self.contents = self.load_dataset(mode)
    
    def load_dataset(self, mode):
        if mode == 'train':
            path = self.config.train_path
        elif mode == 'dev':
            path = self.config.dev_path
        elif mode == 'test':
            path = self.config.test_path
        else :
            raise NotImplementedError(f'{mode} should be in [train, dev or test].')
        with open(path, mode='r', encoding='utf-8') as f:
            contents = f.readlines()
        return contents
    
    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, index) :
        line_items = self.contents[index].split(' ')
        seq1, seq2, label = line_items
        seq = seq1 + SEP + seq2

        encodings = self.config.tokenizer.encode_plus(
            seq, padding='max_length', truncation=True, max_length=self.config.pad_size, return_tensors='pt')
        return encodings.input_ids.squeeze(0), \
            encodings.attention_mask.squeeze(0), \
                encodings.token_type_ids.squeeze(0), \
                    torch.LongTensor([int(label)])

class BertTrainer(object):
    def __init__(self, config:Config, model:Model):
        self.dataloaders = {
            x : DataLoader(
                BertDataset(config, mode=x),
                batch_size=config.batch_size, 
                shuffle=True if x == 'train' else False, 
            )
            for x in ['train', 'dev', 'test']
        }
        self.num_epochs = config.num_epochs
        self.val_steps = config.val_steps
        self.save_path = config.save_path
        
        self.model = model.to(config.device)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        self.total_steps = 0
        self.dev_best_acc = 0.
        self.test_best_acc = 0.
    
    def save_model(self, name='best_test.pt'):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name))
    
    def log(self, msg):
        with open(os.path.join(self.save_path, 'log.txt'), mode='w+') as f:
            f.write(f'{msg}\n')
        return msg
    
    @torch.enable_grad()
    def train(self):
        for epoch in range(self.num_epochs):
            print(self.log('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs)))
            for input_ids, attention_mask, token_type_ids, labels in tqdm(self.dataloaders['train']):
                self.model.train()
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = F.cross_entropy(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.total_steps % self.val_steps == 0 and self.total_steps != 0:
                    improve = ''
                    dev_acc, dev_loss =  self.evaluate(self.dataloaders['dev'])
                    if dev_acc > self.dev_best_acc :
                        self.dev_best_acc = dev_acc
                        self.save_model('best_dev.pt')
                        improve += '*'
                    test_acc, test_loss = self.evaluate(self.dataloaders['test'])
                    if test_acc > self.test_best_acc :
                        self.test_best_acc = test_acc
                        self.save_model('best_test.pt')
                        improve += '^'
                    msg = ('Iter: {0:>6}, Dev Loss: {1:>5.2}, Dev Acc: {2:>6.2%}, Test Loss: {3:>5.2}, Test Acc: {4:>6.2%} {5}')
                    msg = msg.format(self.total_steps, dev_loss, dev_acc, test_loss, test_acc, improve)
                    print(self.log(msg))
                    self.model.train()
                
                self.total_steps += 1
    
    @torch.no_grad()
    def evaluate(self, data_loader:DataLoader):
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in data_loader:
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predict = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predict)

        acc = metrics.accuracy_score(labels_all, predict_all)
        return acc, loss_total / len(data_loader)