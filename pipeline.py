import torch

from models.ERNIE import Config, Model

PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号

if __name__ == '__main__':
    dataset = ''
    checkpoint = ''
    device = 'cpu'
    text1 = ''
    text2 = ''
    pad_size = 128
    
    config = Config()
    model = Model(config)
    
    # loading model...
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    
    # loading data...
    token1 = config.tokenizer.tokenize(text1.strip())
    token2 = config.tokenizer.tokenize(text2.strip())
    token = [CLS] + token1 + [SEP] + token2
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
    
    x = torch.LongTensor(token_ids).to(device)
    seq_len = torch.LongTensor(seq_len).to(device)
    mask = torch.LongTensor(mask).to(device)
    sample = (x, seq_len, mask)
    
    # computing...
    outputs = model(sample)
    predict = torch.argmax(outputs, dim=1).cpu()
    print(predict)