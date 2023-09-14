from transformers import BertModel, BertTokenizer

if __name__ == '__main__':
    model_path = './bert_pretrain/'
    tokenizer:BertTokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    
    text = '你好，Transformers！'
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    outputs = model(**inputs)
    
    print(outputs.attentions[0].shape)
    # attention_weights = outputs.attentions
    # print(attention_weights.shape)