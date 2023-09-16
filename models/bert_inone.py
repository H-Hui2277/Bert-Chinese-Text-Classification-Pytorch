# coding: UTF-8
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW


class Config(object):
    def __init__(self, dataset, device='cuda', num_epochs=3, val_steps=1000, 
                 batch_size=32, pad_size=128, learning_rate=1e-5, 
                 bert_path='./bert_pretrain/', hidden_layer_dropout=0.2, hidden_size=768):
        """ 配置参数
            ---------
            dataset 数据集路径，数据集架构如下
            - dataset 
                - data
                    - class.txt
                    - train.txt
                    - dev.txt
                    - text.txt
            device cuda or cpu \n
            num_epochs 训练轮数 \n
            val_steps 每隔val_steps验证&测试一次 \n
            batch_size 批量大小 \n
            pad_size 句子裁剪长度 \n
            learning_rate 学习率 \n
            bert_path 预训练bert模型地址 \n
            hidden_layer_dropout 输出层的dropout率 \n
            hidden_size bert输出中pooler_output的最后维度数[根据预训练模型确定] \n
        """
        self.model_name = 'BERT'
        self.train_path = os.path.join(dataset, 'data', 'train.txt')
        self.dev_path = os.path.join(dataset, 'data', 'dev.txt')
        self.test_path = os.path.join(dataset, 'data', 'test.txt')
        class_list_path = os.path.join(dataset, 'data', 'class.txt')
        self.class_list = [x.strip() for x in open(
            class_list_path, encoding='utf-8').readlines()]
        self.save_path = os.path.join(dataset, f'saved_dict_{time.strftime("%m%d-%H%M%S")}')
        self.log_file = os.path.join(self.save_path, 'log.txt')
        self.device = torch.device(device)

        self.num_classes = len(self.class_list) 
        self.num_epochs = num_epochs 
        self.val_steps = val_steps 
        
        self.batch_size = batch_size 
        self.pad_size = pad_size 
        self.learning_rate = learning_rate 
        self.bert_path = bert_path
        self.tokenizer:BertTokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_layer_dropout = hidden_layer_dropout
        self.hidden_size = hidden_size


class Model(nn.Module):

    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, output_attentions=True)
        self.dropout = nn.Dropout(config.hidden_layer_dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False):
        """ input_ids 输入句子ids \n
            attention_mask 权重掩码 \n
            token_type_ids Token ids \n
            output_attentions 返回注意力权重[所有layers] \n
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        if output_attentions :
            return out, outputs.attentions
        return out


PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号

    
class BertDataset(Dataset):
    def __init__(self, config:Config, mode='train'):
        super().__init__()
        self.config = config
        self.contents = self.load_dataset(mode)
    
    def load_dataset(self, mode):
        """ 加载数据集
        """
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
            seq, padding='max_length', truncation=True, max_length=self.config.pad_size, return_tensors='pt').to(self.config.device)
        return encodings.input_ids.squeeze(0), \
            encodings.attention_mask.squeeze(0), \
                encodings.token_type_ids.squeeze(0), \
                    torch.LongTensor([int(label)]).to(self.config.device)

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
        """ 保存模型 至以下路径
            - save_path / name
        """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name))
    
    def log(self, msg):
        with open(os.path.join(self.save_path, 'log.txt'), mode='w+') as f:
            f.write(f'{msg}\n')
        return msg
    
    @torch.enable_grad()
    def train(self):
        """ 训练
        """
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
        """ 验证/测试
        """
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
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
    

class BertPipeline(object):
    def __init__(self, config:Config, model_path:str):
        config = config
        self.model = Model(config)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint).to(config.device)
        self.model.eval()
        
        self.tokenizer = config.tokenizer
        self.class_list = config.class_list
    
    @torch.no_grad()
    def __call__(self, text, topk=1):
        inputs = self.tokenizer.encode_plus(text)
        output = self.model(**inputs)
        
        predict_class = []
        indices = torch.topk(output, k=topk, dim=1)
        for indice in indices.indices[0]:
            predict_class.append(self.class_list[indice.item()])
        
        if topk == 1:
            return predict_class[0]
        return predict_class