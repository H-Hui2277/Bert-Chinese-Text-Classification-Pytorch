# coding: UTF-8
import time
import pickle
import os
import re
import random
from datetime import timedelta
from collections import Counter

import torch
import pandas as pd
import jieba
from tqdm import tqdm


PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号

def load_dataset(path, config, pre_loaded=True):
    contents = []
    pre_load_path = path.replace('.txt', '.pkl')
    if pre_loaded:
        if os.path.exists(pre_load_path):
            with open(pre_load_path, mode='rb') as f:
                contents = pickle.load(f)
                f.close()
            print(f'Get the data from {pre_load_path}')
            return contents
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            lin_items = lin.split(' ')
            if len(lin_items) == 2:
                content, label = lin_items
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
            elif len(lin_items) == 3:
                content1, content2, label = lin_items
                token1 = config.tokenizer.tokenize(content1)
                token2 = config.tokenizer.tokenize(content2)
                token = [CLS] + token1 + [SEP] + token2
            else:
                raise ValueError(f'{lin_items} is supposed to have 2 or 3 items.')
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            if config.pad_size:
                if len(token) < config.pad_size:
                    mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
                    token_ids += ([0] * (config.pad_size - len(token)))
                else:
                    mask = [1] * config.pad_size
                    token_ids = token_ids[:config.pad_size]
                    seq_len = config.pad_size
            contents.append((token_ids, int(label), seq_len, mask))
    if pre_loaded:
        with open(pre_load_path, mode='wb') as f:
            pickle.dump(contents, f)
            f.close()
        print(f'Save data to {pre_load_path}')
    return contents

def build_dataset(config):
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)
    return train, dev, test


def get_class_balanced_weight(config):
    """ 计算类别平衡权重
    """
    print('computing class balanced weight...')
    contents = load_dataset(config.train_path, config)
    class_count = {i:0 for i in range(len(config.class_list))}
    for x, y, seq_len, mask in contents:
        class_count[y] += 1
    count_list = torch.Tensor([v for k, v in class_count.items()]).sqrt()
    return count_list.sum() / (len(config.class_list) * count_list)


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


"""字符串预处理工具方法"""
def remove_punctutation(text):
    ''' 将给定字符串中的非字母数字字符和空白字符删除
    '''
    return re.sub('[\W\s]', '', text)

def get_pattern(stop_words_file, encoding='utf-8'):
    ''' 读取停用词表\n
    stop_words_file 停用词表路径\n
    return 停用词表的正则表达式\n
    '''
    with open(stop_words_file, mode='r', encoding=encoding) as f:
        words = f.readlines()
        f.close()
    pa_text = ''
    for word in words:
        if remove_punctutation(word).strip() == '':
            continue
        pa_text = f'{word.strip()}|{pa_text}'
    return pa_text

class Reformator(object):
    def __init__(self, remove_punc=True, stop_words_file=None, stop_words_encoding='utf-8', \
        addtional_patterns=None):
        '''
        remove_punc 是否删除字符串中的标点符号\n
        stop_words_file 停用词表路径，为None时不使用停用词表\n
        stop_words_encoding 停用词表的编码格式\n
        return 返回重新编码后的字符串
        '''
        self.remove_punc = remove_punc
        self.pattern = get_pattern(stop_words_file, stop_words_encoding) \
            if stop_words_file is not None else None
        self.aps = addtional_patterns
            
    def __call__(self, text:str):
        ''' text 输入中文字符串\n
        return 重新编码后的字符串\n
        '''
        text = str(text)
        if self.remove_punc:
            text = remove_punctutation(text)
        if self.pattern is not None:
            text = re.sub(self.pattern, '', text)
        if self.aps is not None:
            for ap in self.aps:
                text = re.sub(ap, '', text)
        return text

""""高低频词相关方法"""
def get_freq_words(text, k=5):
    ''' 获取一段文本中的高频词或低频词\n
        text: 输入文本 \n
        k: 前k个高频词或低频词 \n
        return 高低频词列表
    '''
    words = jieba.cut(text)
    words_count = Counter(words)
    most_common = words_count.most_common()
    
    high_freq_words = [word for word, count in most_common[:k]]
    low_freq_words = [word for word, count in most_common[-k:]]
    
    return high_freq_words, low_freq_words

def get_freq_words_from_file(file, encoding='utf-8', k=5, save_file=None):
    ''' 获取一个文本文件中的高频词或低频词 \n
        file 文本文件路径 \n
        encoding 文本文件编码格式 
    '''
    with open(file, mode='r', encoding=encoding) as f:
        text = f.read()
        f.close()
    
    text = re.sub('[\s]', '', text) # 去除空白字符，包括空格、回车符等
    high_freq_words, low_freq_words = get_freq_words(text.strip(), k)
    if save_file is not None:
        with open(save_file, mode='w', encoding='utf-8') as f:
            for word in high_freq_words + low_freq_words:
                f.write(f'{word}\n')
            f.close()
    return high_freq_words, low_freq_words

"""从原始数据文件中直接构建数据集"""
def dataset_transform(origin_file, save_dir, train_rate=0.8, seed=1108, pre_loading=True,
                  remove_punc=True, stop_words_file=None, stop_words_file_encoding='utf-8', addtional_patterns=None, ):
    """ 从原始数据文件中构建数据集 \n
        origin_file 原始数据文件地址 \n
        save_dir 数据集保存地址 \n
        train_rate 训练集占比 \n
        seed 固定随机数种子，使每次划分的结果保持一致 \n
        pre_loading 预存原始数据为二进制数据，加快后续读取速度 \n
        else Reformator参数
    """
    random.seed(seed)
    print('loading data...')
    start_time = time.time()
    pre_loading_file = origin_file.replace('.xlsx', '.pkl')
    if pre_loading and os.path.exists(pre_loading_file):
        with open(pre_loading_file, mode='rb') as f:
            frame = pickle.load(f)
    else:
        frame = pd.read_excel(origin_file, usecols=['接收单位', '案件类型', '来电内容'], dtype=str)
        if pre_loading:
            with open(pre_loading_file, mode='wb') as f:
                pickle.dump(frame, f)
    cost = get_time_dif(start_time)
    print(f'loading cost {cost} s.')
    
    class_list = set(frame['接收单位'])
    class_dict = {}
    with open(os.path.join(save_dir, 'class.txt'), mode='w+', encoding='utf-8') as f:
        for i, cls in enumerate(class_list):
            class_dict[str(cls)] = i
            f.write(f'{cls}\n')
    
    dlen = len(frame['案件类型'])
    indices = [i for i in range(dlen)]
    random.shuffle(indices)
    train_len = int(dlen * train_rate)
    dev_len = int(dlen * (1 - train_rate) / 2.)
    
    train_file = open(os.path.join(save_dir, 'train.txt'), mode='w+', encoding='utf-8')
    dev_file = open(os.path.join(save_dir, 'dev.txt'), mode='w+', encoding='utf-8')
    test_file = open(os.path.join(save_dir, 'test.txt'), mode='w+', encoding='utf-8')
    
    reformator = Reformator(remove_punc, stop_words_file, stop_words_file_encoding, addtional_patterns)
    
    print('dumping data...')
    start_time = time.time()
    for i, indice in tqdm(enumerate(indices)):
        text1 = reformator(frame['案件类型'][indice])
        text2 = reformator(frame['来电内容'][indice])
        label = class_dict[str(frame['接收单位'][indice])]
        if i <= train_len:
            train_file.write(f'{text1} {text2} {label}\n')
        elif i <= train_len + dev_len:
            dev_file.write(f'{text1} {text2} {label}\n')
        else :
            test_file.write(f'{text1} {text2} {label}\n')
    train_file.close()
    dev_file.close()
    test_file.close()
    cost = get_time_dif(start_time)
    print(f'dumping cost {cost} s.')

def top_k_accuracy(y_true, y_pred, k=5):
    """ Top-k 精度 \n
        y_true 真实的目标标签 \n
        y_pred 模型预测的目标标签概率分布
    """
    _, pred_indices = torch.topk(y_pred, k, dim=1)
    correct = torch.sum(torch.eq(pred_indices, y_true.view(-1, 1)).any(dim=1))
    accuracy = correct.item() / len(y_true)
    return accuracy