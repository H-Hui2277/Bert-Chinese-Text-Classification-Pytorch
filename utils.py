# coding: UTF-8
import time
from datetime import timedelta
import pickle
import os
import re

import torch
from tqdm import tqdm


PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32, pre_loaded=True):
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

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        if pre_loaded:
            with open(pre_load_path, mode='wb') as f:
                pickle.dump(contents, f)
                f.close()
            print(f'Save data to {pre_load_path}')
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


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
    ''' 删除字符串中标点符号
    '''
    return re.sub('[^\w\s]', '', text)

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
        text = text.strip()
        if self.remove_punc:
            text = remove_punctutation(text)
        if self.pattern is not None:
            text = re.sub(self.pattern, '', text)
        if self.aps is not None:
            for ap in self.aps:
                text = re.sub(ap, '', text)
        return text
