from transformers import BertModel, BertTokenizer


PAD, CLS, SEP, UNK = '[PAD]', '[CLS]', '[SEP]', '[UNK]'  # padding符号, bert中综合信息符号


if __name__ == '__main__':
    model_path = './ERNIE_pretrain/'
    tokenizer:BertTokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    
    text =  ['你好', '第二个句子']
    inputs = tokenizer.encode_plus(text, padding='max_length', max_length=8, truncation=True, return_tensors='pt')
    print(inputs.input_ids)
    print(inputs.token_type_ids)
    print(inputs.attention_mask)
    outputs = model(**inputs)
    
    print(outputs.pooler_output.shape)
    # print(outputs.attentions[0].shape)
    # attention_weights = outputs.attentions
    # print(attention_weights.shape)