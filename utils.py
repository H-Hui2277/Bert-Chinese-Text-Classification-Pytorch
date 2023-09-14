# coding: UTF-8
import time
import pickle
import os
import re
import random
from datetime import timedelta
from collections import Counter
import chardet
from chardet.universaldetector import UniversalDetector

import torch
import pandas as pd
import jieba
from tqdm import tqdm

PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号


def get_file_encoding(file_path):
    """ 获取文本文件编码 [全部读取] \n
        file_path 文件路径 \n
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        encoding = chardet.detect(raw_data).get('encoding')
    return encoding


def get_file_encoding_detector(file_path):
    """ 获取文本文件编码 [部分读取] \n
        file_path 文件路径 \n
    """
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']


def load_dataset(path, config, get_contents_from_presaved_file=True):
    contents = []
    presaved_file_path = path.replace('.txt', '.pkl')
    if get_contents_from_presaved_file and os.path.exists(presaved_file_path):
        with open(presaved_file_path, mode='rb') as f:
            contents = pickle.load(f)
            f.close()
        print(f'Get the data from {presaved_file_path}')
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
            contents.append((token_ids, int(label), seq_len, mask))
    with open(presaved_file_path, mode='wb') as f:
        pickle.dump(contents, f)
        f.close()
    print(f'Presave data to {presaved_file_path}')
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
    class_count = {i: 0 for i in range(len(config.class_list))}
    for x, y, seq_len, mask in contents:
        class_count[y] += 1
    count_list = torch.Tensor([v for k, v in class_count.items()]).sqrt()
    return count_list.sum() / (len(config.class_list) * count_list)


class DatasetIterator(object):
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
    return DatasetIterator(dataset, config.batch_size, config.device)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


"""字符串预处理工具方法"""


class RegexDeletionTool(object):
    def __init__(self, remove_punctuations, remove_numbers, remove_characters):
        """ 正则删除工具 \n
            remove_punctuations 删除非字符数字的字符 \n
            remove_numbers 删除数字 \n
            remove_characters 删除字母 \n
        """
        self._rp = remove_punctuations
        self._rn = remove_numbers
        self._rc = remove_characters

    def __call__(self, text):
        if self._rp:
            text = re.sub('[\W\s]', '', text)
        if self._rn:
            text = re.sub(r'[0-9]', '', text)
        if self._rc:
            text = re.sub(r'[a-zA-Z]', '', text)
        return text


def get_pattern(stop_words_file):
    """ 读取停用词表\n
    stop_words_file 停用词表路径\n
    return 停用词表的正则表达式\n
    """
    encoding = get_file_encoding_detector(stop_words_file)
    with open(stop_words_file, mode='r', encoding=encoding) as f:
        words = f.readlines()
        f.close()
    pa_text = None
    for word in words:
        reword = re.sub('[\W\s]', '', word)
        if reword == '':
            continue
        pa_text = f'{reword}|{pa_text}' if pa_text is not None else f'{reword}'
    return pa_text


class Reformator(object):
    def __init__(self, remove_punc=True, remove_numbers=True, remove_characters=True,
                 stop_words_file=None, additional_patterns=None):
        """
        remove_punc 是否删除字符串中的标点符号\n
        stop_words_file 停用词表路径，为None时不使用停用词表\n
        stop_words_encoding 停用词表的编码格式\n
        return 返回重新编码后的字符串
        """
        self.regex_tool = RegexDeletionTool(remove_punc, remove_numbers, remove_characters)
        self.pattern = get_pattern(stop_words_file) \
            if stop_words_file is not None else None
        self.aps = additional_patterns

    def __call__(self, text: str):
        """ text 输入中文字符串\n
        return 重新编码后的字符串\n
        """
        text = str(text)
        text = self.regex_tool(text)
        if self.pattern is not None:
            text = re.sub(self.pattern, '', text)
        if self.aps is not None:
            for ap in self.aps:
                text = re.sub(ap, '', text)
        return text


""""高低频词相关方法"""


def get_freq_words(text, k=5):
    """ 获取一段文本中的高频词或低频词\n
        text: 输入文本 \n
        k: 前k个高频词或低频词 \n
        return 高低频词列表
    """
    words = jieba.cut(text)
    words_count = Counter(words)
    most_common = words_count.most_common()

    high_freq_words = [word for word, count in most_common[:k]]
    low_freq_words = [word for word, count in most_common[-k:]]

    return high_freq_words, low_freq_words


def get_freq_words_from_file(file, encoding='utf-8', k=5, save_file=None):
    """ 获取一个文本文件中的高频词或低频词 \n
        file 文本文件路径 \n
        encoding 文本文件编码格式
    """
    with open(file, mode='r', encoding=encoding) as f:
        text = f.read()
        f.close()

    text = re.sub('\s', '', text)  # 去除空白字符，包括空格、回车符等
    high_freq_words, low_freq_words = get_freq_words(text.strip(), k)
    if save_file is not None:
        with open(save_file, mode='w', encoding='utf-8') as f:
            for word in high_freq_words + low_freq_words:
                f.write(f'{word}\n')
            f.close()
    return high_freq_words, low_freq_words


"""从原始数据文件中直接构建数据集"""


def dataset_transform(origin_file, save_dir, train_rate=0.8, seed=1108, get_contents_from_presaved_file=True,
                      remove_punc=True, remove_numbers=True, remove_characters=True,
                      stop_words_file=None, remove_high_and_low_freq_words=True, remove_from_presaved_file=True, k=5):
    """ 
        从原始数据文件中构建数据集，建议构造如下
        - Dataset
            - data.xlsx 原始数据文件
            - data 处理后的数据集保存地址
            
        origin_file 原始数据文件地址 \n
        save_dir 数据集保存地址 \n
        train_rate 训练集占比 \n
        seed 固定随机数种子，使每次划分的结果保持一致 \n
        get_contents_from_presaved_file 预存原始数据为二进制数据，加快后续读取速度 \n
        remove_punc 删除字母数字外的标点符号 \n
        remove_numbers 删除数字 \n
        remove_characters 删除字母 \n
        stop_words_file 停用词表 \n

        remove_high_and_low_freq_words 删除高频词和低频词 NOTE [未经过测试，效果不确定] \n
        remove_from_presaved_file 从预存的高低频词文件中删除高低频词 \n
        k 分别删除高、低频词的个数 \n
    """
    random.seed(seed)
    print('loading data...')
    start_time = time.time()
    presaved_file = origin_file.replace('.xlsx', '.pkl')
    if get_contents_from_presaved_file and os.path.exists(presaved_file):
        with open(presaved_file, mode='rb') as f:
            frame = pickle.load(f)
    else:
        frame = pd.read_excel(origin_file, usecols=['接收单位', '案件类型', '来电内容'], dtype=str)
        if get_contents_from_presaved_file:
            with open(presaved_file, mode='wb') as f:
                pickle.dump(frame, f)
    cost = get_time_dif(start_time)
    print(f'loading cost {cost} s.')

    class_list = set(frame['接收单位'])
    class_dict = {}
    with open(os.path.join(save_dir, 'class.txt'), mode='w+', encoding='utf-8') as f:
        for i, cls in enumerate(class_list):
            class_dict[str(cls)] = i
            f.write(f'{cls}\n')

    if remove_high_and_low_freq_words:
        print('loading high and low freq words')
        start_time = time.time()
        words_presaved_file = os.path.join(save_dir, 'high_low_freq_words.pkl')
        if remove_from_presaved_file and os.path.exists(words_presaved_file):
            print(f'loading from presaved file {words_presaved_file}')
            with open(words_presaved_file, mode='rb') as f:
                high_low_freq_words = pickle.load(f)
        else:
            words_counter = Counter()
            regex_deletion_tool = RegexDeletionTool(True, True, True)
            for seq in frame['来电内容']:
                words = jieba.cut(regex_deletion_tool(seq))
                words_counter.update(words)
            most_common = words_counter.most_common()
            high_freq_words = [word for word, count in most_common[:k]]
            low_freq_words = [word for word, count in most_common[-k:]]
            print(f'pre saved words to file {words_presaved_file}')
            with open(words_presaved_file, mode='wb') as f:
                high_low_freq_words = low_freq_words + high_freq_words
                pickle.dump(high_low_freq_words, f)

        print(f'high and low freq words list: {high_low_freq_words}')
        additional_patterns = None
        for word in high_low_freq_words:
            additional_patterns = f'{word}|{additional_patterns}' \
                if additional_patterns is not None else f'{word}'
        print(additional_patterns)
        print(f'loading high and low freq words cost {get_time_dif(start_time)} s.')
    else:
        additional_patterns = None

    dlen = len(frame['案件类型'])
    indices = [i for i in range(dlen)]
    random.shuffle(indices)
    train_len = int(dlen * train_rate)
    dev_len = int(dlen * (1 - train_rate) / 2.)

    train_file = open(os.path.join(save_dir, 'train.txt'), mode='w+', encoding='utf-8')
    dev_file = open(os.path.join(save_dir, 'dev.txt'), mode='w+', encoding='utf-8')
    test_file = open(os.path.join(save_dir, 'test.txt'), mode='w+', encoding='utf-8')

    reformator = Reformator(remove_punc, remove_numbers, remove_characters,
                            stop_words_file, additional_patterns)

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
        else:
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
