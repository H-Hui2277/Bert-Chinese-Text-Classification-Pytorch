# %%
import torch

from models.bert_inone import Config, Model, BertDataset

config = Config('./Dataset/')
config.device = 'cpu'

model = Model(config)

text = '你好，这是一段测试语句'
inputs = config.tokenizer.encode_plus(text, return_tensors='pt')
outputs, attentions = model(**inputs, output_attentions=True)
print(attentions[-1].shape)

# %%
import os 

import numpy as np
from matplotlib import pyplot as plt 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

attention_maps = attentions[-1].squeeze(0).detach().numpy()
fig, ax = plt.subplots(3, 4)
for i in range(len(attention_maps)) :
    ax[i//4, i%4].imshow(attention_maps[i]/np.sum(attention_maps[i], axis=-1, keepdims=True), cmap='hot')
    # ax[i//4, i%4].set_xticks(np.arange(attention_maps[i].shape[1]))
    # ax[i//4, i%4].set_yticks(np.arange(attention_maps[i].shape[0]))
plt.tight_layout()
plt.show()


