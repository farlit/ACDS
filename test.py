import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import config
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from model import BertSegPos
from evaluate import evaluate
from data_process import load_data
from test_file import generate_file
from transformers import AutoModel
from torch.utils.data import DataLoader
from data_loader import AnChinaDataset
# from metrics import f1_score, bad_case, output_write, output2res
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
sentences,seg,pos,segpos,flag,gram_list,positions,gram_maxlen,gram2id=load_data('./EvaHan_testb_gold.txt')
test_size=len(sentences)//10
train_size=len(sentences)-2*test_size

test_sentences=sentences[train_size+test_size:]
test_seg=seg[train_size+test_size:]
test_pos=pos[train_size+test_size:]
test_segpos=segpos[train_size+test_size:]
test_gram_list=gram_list[train_size+test_size:]
test_positions=positions[train_size+test_size:]
test_gram_maxlen=gram_maxlen[train_size+test_size:]
test_flags = flag[train_size+test_size:]
# generate_file(test_sentences,test_seg,test_pos,'temp.txt',flag[train_size+test_size:])
test_dataset =AnChinaDataset(test_sentences, test_seg, test_pos,test_segpos,test_gram_list,test_positions,test_gram_maxlen,gram2id,test_flags)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')
test_loader  = DataLoader(test_dataset, batch_size=config.batch_size,\
                          collate_fn=test_dataset.collate_fn,num_workers=0,pin_memory=True)
model=BertSegPos(config,gram2id)
# print(list(model.parameters())[-8])
# print(list(model.parameters())[-7])
model.to(device)
pre_model = torch.load('model_base1.pth')
now_model_dict = model.state_dict()
state_dict = {k:v for k,v in pre_model.items() if k in now_model_dict.keys()}
now_model_dict.update(state_dict)
model.load_state_dict(now_model_dict)
result=evaluate(test_loader, model, 'eval')
print("test:\n result:{}".format(result))
