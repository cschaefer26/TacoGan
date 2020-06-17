import numpy as np

from text.text_cleaner import basic_cleaners
from text.tokenizer import Tokenizer
from utils.config import Config

target = np.load('/tmp/target_1000.npy')
pred = np.load('/tmp/pred_1000.npy')

cfg = Config.load('config.yaml')
tokenizer = Tokenizer(basic_cleaners, cfg.symbols)

target_len = target.shape[0]
pred_len = pred.shape[0]

pred_max = np.zeros((pred_len, target_len))
for i in range(pred_len):
    pred_max[i] = pred[i, target]

print(pred)
#print(pred_max[:10, :10])

#print(target)
#print(tokenizer.decode(target.tolist()))
#print(pred)