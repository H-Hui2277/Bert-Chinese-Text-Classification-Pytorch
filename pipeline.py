import torch

from models.ERNIE import Config, Model
from utils import Reformator

PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号

class Predictor(object):
    def __init__(self, dataset='./Dataset_baidu/', checkpoint='./Dataset_baidu/saved_dict_0909-213159/best_test.pt', \
                 device='cuda', pad_size=128, remove_punc=True, stop_words_file=None, stop_words_file_encoding='utf-8'):
        self.device=device
        self.config = Config(dataset)
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model.eval().to(self.device)

        self.pad_size = pad_size
        self.reformator = Reformator(remove_punc, stop_words_file, stop_words_file_encoding)
    
    def __call__(self, text1:str, text2:str):
        text1, text2 = self.reformator(text1), self.reformator(text2)
        token1 = self.config.tokenizer.tokenize(text1)
        token2 = self.config.tokenizer.tokenize(text2)
        token = [CLS] + token1 + [SEP] + token2
        seq_len = len(token)
        mask = []
        token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
        if self.pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * self.pad_size
                token_ids = token_ids[:self.pad_size]
                seq_len = self.pad_size
        x = torch.LongTensor(token_ids).view(1, -1).to(self.device)
        seq_len = torch.LongTensor(seq_len).view(1, -1).to(self.device)
        mask = torch.LongTensor(mask).view(1, -1).to(self.device)
        sample = (x, seq_len, mask)

        outputs = self.model(sample)
        predict = torch.argmax(outputs, dim=1).cpu()
        predict_class = self.config.class_list[predict.item()]
        return predict_class

if __name__ == '__main__':
    dataset = './Dataset_baidu/'
    checkpoint = './Dataset_baidu/saved_dict_0909-213159/best_test.pt'
    device = 'cuda'
    text1 = '邮政市场监管'
    text2 = '市民来电其有一个申通西安寄往长沙月日西安发出月日长沙转运中心更新物流快递方反馈未果来电请相关单位核实快递物流长时间未更新'
    pad_size = 128
    remove_punc = True
    stop_words_file = None

    predictor = Predictor(dataset=dataset, checkpoint=checkpoint, device=device, pad_size=pad_size, \
                          remove_punc=remove_punc, stop_words_file=stop_words_file)
    
    predict_class = predictor(text1, text2)
    print(predict_class)