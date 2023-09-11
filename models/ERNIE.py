# coding: UTF-8
import time
import os

import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.save_path = dataset + f'/saved_dict_{time.strftime("%m%d-%H%M%S")}/'    # 模型训练结果
        self.log_file = os.path.join(self.save_path, 'log.txt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5                                       # 学习率
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_layer_dropout = 0.2
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config:Config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.hidden_layer_dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            m.requires_grad_(False)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out
