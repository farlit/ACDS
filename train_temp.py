import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
from transformers import AutoModel
from torch.utils.data import DataLoader
from data_loader import AnChinaDataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
# from metrics import f1_score, bad_case, output_write, output2res
from transformers.optimization import get_cosine_schedule_with_warmup
sentences,seg,pos,segpos,flag,gram_list,positions,gram_maxlen,gram2id=load_data('./zuozhuan_train_utf8.txt')
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
length = len(sentences)
part_length = length//5
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()
args.world_size = int((len(os.environ["CUDA_VISIBLE_DEVICES"])+1)/2)
config.learning_rate *= args.world_size

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

model=BertSegPos(config, gram2id)
model.to(device)
i = 0
index_vaild = [True if i*part_length<=j<(i+1)*part_length else False for j in range(length)]
index_train = [False if index else True for index in index_vaild]
train_sentences=np.array(sentences,dtype=object)[index_train]
train_seg=np.array(seg,dtype=object)[index_train]
train_pos=np.array(pos,dtype=object)[index_train]
train_segpos=np.array(segpos,dtype=object)[index_train]
if config.use_attention:
    train_gram_list=np.array(gram_list,dtype=object)[index_train]
    train_positions=np.array(positions,dtype=object)[index_train]
    train_gram_maxlen=np.array(gram_maxlen,dtype=object)[index_train]
else:
    train_gram_list=None
    train_positions=None
    train_gram_maxlen=None

val_sentences=np.array(sentences,dtype=object)[index_vaild]
val_seg=np.array(seg,dtype=object)[index_vaild]
val_pos=np.array(pos,dtype=object)[index_vaild]
val_segpos=np.array(segpos,dtype=object)[index_vaild]
if config.use_attention:
    val_gram_list=np.array(gram_list,dtype=object)[index_vaild]
    val_positions=np.array(positions,dtype=object)[index_vaild]
    val_gram_maxlen=np.array(gram_maxlen,dtype=object)[index_vaild]
else:
    val_gram_list=None
    val_positions=None
    val_gram_maxlen=None

train_dataset=AnChinaDataset(train_sentences,train_seg,train_pos,train_segpos,train_gram_list,train_positions,train_gram_maxlen,gram2id)
if args.local_rank == 0:
    val_dataset=AnChinaDataset(val_sentences,val_seg,val_pos,val_segpos,val_gram_list,val_positions,val_gram_maxlen,gram2id)
# test_dataset =AnChinaDataset(test_sentences, test_seg, test_pos,test_segpos,test_gram_list,test_positions,test_gram_maxlen,gram2id)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size,sampler=DistributedSampler(train_dataset),\
                         collate_fn=train_dataset.collate_fn,num_workers=os.cpu_count(),pin_memory=True)
if args.local_rank == 0:
    val_loader  = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=val_dataset.collate_fn,\
                             num_workers=os.cpu_count(),pin_memory=True)

if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
#   if config.load_before:
#       map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
#       model.load_state_dict(torch.load('sikuRoberta_model.pth', map_location=map_location))

if args.local_rank == 0:
    if config.load_before:
        # model.load_state_dict(torch.load('model_retry_alldata.pth'))
        pre_model = torch.load('model_aug_trans1.pth')
        now_model_dict = model.state_dict()
        state_dict = {k:v for k,v in pre_model.items() if k in ['transform2noise_seg.weight','transform2noise_pos.weight']}
        now_model_dict.update(state_dict)
        model.load_state_dict(now_model_dict)
    torch.distributed.barrier()   # Make sure only the first process in distributed training will download model & vocab
print(list(model.parameters())[-8])
if config.full_fine_tuning:
    # optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False, no_deprecation_warning=True)
    # transform2noise_seg_params = list(map(id, model.transform2noise_seg.parameters()))
    # transform2noise_pos_params = list(map(id, model.transform2noise_pos.parameters()))
    # base_params = filter(lambda p: id(p) not in transform2noise_seg_params + transform2noise_pos_params,
    #                      model.parameters())
    # optimizer = torch.optim.AdamW([{'params': base_params},
    #     {'params': model.transform2noise_seg.parameters(), 'lr': config.learning_rate * 1000},
    #     {'params': model.transform2noise_pos.parameters(), 'lr': config.learning_rate * 1000}], lr=config.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
else:
    freeze_layers = ['transform2noise_seg.weight', 'transform2noise_pos.weight']
    for name, param in model.named_parameters():
        param.requires_grad = True
        for unactive in freeze_layers:
            if unactive in name:
                param.requires_grad = False
                break
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

train_size=len(train_dataset)
train_steps_per_epoch = train_size // config.batch_size
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_steps_per_epoch,
                                        num_training_steps=config.epoch_num * train_steps_per_epoch)
model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
best_val_f1 = 0.0
patience_counter = 0
for epoch in range(1, config.epoch_num + 1):
    train_losses = 0
    model.train()
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batchseg_labels, batchpos_labels, batchsegpos_labels,\
        batchgram_list, matching_matrix, channel_ids, _, _0 = batch_samples
        # shift tensors to GPU if available
        batch_data = batch_data.to(device)
        batch_seglabels = batchseg_labels.to(device)
        batch_poslabels = batchpos_labels.to(device)
        batch_segposlabels = batchsegpos_labels.to(device)
        batch_masks = batch_data.gt(0).to(device)  # get padding mask
        batch_gramlist = batchgram_list.to(device)
        matching_matrix = matching_matrix.to(device)
        channel_ids = channel_ids.to(device)
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, seglabels=batch_seglabels,\
                     poslabels=batch_poslabels, segposlabels=batch_segposlabels, gram_list=batch_gramlist,\
                     matching_matrix=matching_matrix, channel_ids=channel_ids)[0]
        train_losses += float(loss.item())
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
        if args.local_rank==0 and (idx+1)%20==0:
            print(list(model.parameters())[-8])
            print("Epoch: {}, batch:{}, train loss: {}".format(epoch, idx+1, loss.item()))
        torch.cuda.empty_cache()

    if args.local_rank == 0:
        result=evaluate(val_loader, model)
        train_loss = train_losses / len(train_loader)
        print("Epoch: {}, train loss: {}, result: {}\n".format(epoch, train_loss, result))

        val_f1 = result['PoS'][2]
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            #  选择一个进程保存
        # if args.local_rank == 0:
            torch.save(model.module.state_dict(), 'model_base{}_noise.pth'.format(i+1))
            print("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        # print("Saving end!")
        # print(model.device)
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            print("Best val f1: {}".format(best_val_f1))
            break

print("The {}th Training Finished!".format(str(i+1)))

# import os
# def get_exec_out(sxcute_str):
#     out_list = os.popen(sxcute_str).readlines()
#     return out_list
# excute_str = 'fuser -v /dev/nvidia*'
# out_list = get_exec_out(excute_str)
# for oo in out_list:
#     kill_str = 'kill -9 ' + str(oo)
#     os.system(kill_str)