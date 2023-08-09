import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import config
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import BertSegPos
from evaluate_aug_complete import evaluate
from data_process import load_data
from test_file import generate_file
from transformers import AutoModel
from torch.utils.data import DataLoader
from data_loader import AnChinaDataset
# from metrics import f1_score, bad_case, output_write, output2res
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
data_dir = './tgt.shuf.seg_pos'
# data_dir = '/EvaHan_testb_raw.txt'
sentences,segs,poss,segpos,flag,gram_list,positions,gram_maxlen,gram2id=load_data(data_dir)
print("load data success!")
# generate_file(test_sentences,test_seg,test_pos,'temp.txt',flag[train_size+test_size:])
test_dataset =AnChinaDataset(sentences,segs,poss,segpos,gram_list,positions,gram_maxlen,gram2id)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')
test_loader  = DataLoader(test_dataset, batch_size=32,\
                          collate_fn=test_dataset.collate_fn,num_workers=os.cpu_count(),pin_memory=True)
model1=BertSegPos(config,None)
model1.to(device)
model1.load_state_dict(torch.load('4-Fold/model_aug_origin1.pth', map_location="cuda:0"))

model2=BertSegPos(config,None)
model2.to(device)
model2.load_state_dict(torch.load('4-Fold/model_aug_origin2.pth', map_location="cuda:0"))

model3=BertSegPos(config,None)
model3.to(device)
model3.load_state_dict(torch.load('4-Fold/model_aug_origin3.pth', map_location="cuda:0"))

model4=BertSegPos(config,None)
model4.to(device)
model4.load_state_dict(torch.load('4-Fold/model_aug_origin4.pth', map_location="cuda:0"))
print("load model success!")
pred_segs,pred_poss=evaluate(test_loader, model1, model2, model3, model4, 'test')
print("predict finish!")
generate_file(sentences, pred_segs, pred_poss, '4-Fold/tgt.shuf.seg_pos-aug-infer', flag)
print("generate enhanced data success!")
# all_data=open('tgt.shuf.seg_pos-infer-repair', 'r', encoding='utf-8').readlines()
# with open("tgt.shuf.seg_pos-reweight-long", 'a', encoding='utf-8-sig') as f:
#     for data in all_data:
#         if data == '\n':
#             continue
#         data = data[:-1]
#         data += '。/w'
#         f.write(data)
#         f.write('\n')
#         # split_chars = [',',':','，','：','。']
#         # length_data = len(data)
#         # pre_loc = 0
#         # for i in range(length_data):
#         #     if data[i] in split_chars:
#         #         f.write(data[pre_loc:i+3])
#         #         f.write('\n')
#         #         pre_loc = i+4
# f.close()
# all_data=open("tgt.shuf.seg_pos-reweight-long", 'r', encoding='utf-8').readlines()
# with open("tgt.shuf.seg_pos-reweight0-long", 'a', encoding='utf-8-sig') as f:
#     for data in all_data:
#         if data == '\n':
#             continue
#         data = data[:-1]
#         word_tags = data.split(' ')
#         recheck_data = ''
#         for word_tag in word_tags:
#             split_word_tag = word_tag.split('/')
#             if len(split_word_tag) != 2:
#                 continue
#             word = split_word_tag[0]
#             tag = split_word_tag[1]
#             if not word or not tag:
#                 continue
#             if len(word)>=5 or len(tag)>2:
#                 continue
#             if tag[0]<'a' or tag[-1]>'z' or tag[0]<'a' or tag[-1]>'z':
#                 continue
#             recheck_data += word_tag
#             recheck_data += ' '
#         if recheck_data:
#             recheck_data = recheck_data[:-1]
#             recheck_data += '\n'
#         f.write(recheck_data)