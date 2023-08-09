import config
from data_process import load_data
from collections import defaultdict

sentences,seg,pos,segpos,flag,gram_list,positions,gram_maxlen,gram2id=load_data('./zuozhuan_train_utf8.txt')
seg_freqs = defaultdict(int)
pos_freqs = defaultdict(int)
segpos_freqs = defaultdict(int)
for labels in zip(seg,pos,segpos):
    for i in range(len(labels[0])):
        seg_freqs[labels[0][i]] += 1
        pos_freqs[labels[1][i]] += 1
        segpos_freqs[labels[2][i]] += 1

label_seg2id = {}
id_seg2label = {}
uniq_tokens = [token for token, freq in seg_freqs.items()]
for i in range(len(uniq_tokens)):
    label_seg2id[uniq_tokens[i]]=i
    id_seg2label[i]=uniq_tokens[i]
num_seglabels = len(label_seg2id)

label_pos2id = {}
id_pos2label = {}
uniq_tokens = [token for token, freq in pos_freqs.items()]
for i in range(len(uniq_tokens)):
    label_pos2id[uniq_tokens[i]]=i
    id_pos2label[i]=uniq_tokens[i]
num_poslabels = len(label_pos2id)

label_segpos2id = {}
id_segpos2label = {}
uniq_tokens = [token for token, freq in segpos_freqs.items()]
for i in range(len(uniq_tokens)):
    label_segpos2id[uniq_tokens[i]]=i
    id_segpos2label[i]=uniq_tokens[i]
num_segposlabels = len(label_segpos2id)

# train_sentences,train_seg,train_pos,train_segpos,flag,train_gram_list,train_positions,train_gram_maxlen,gram2id=load_data(config.data_dir)
# for i in range(len(train_pos)):
#     label_id = [label_pos2id.get(t) if label_pos2id.get(t) else -1 for t in train_pos[i]]
#     if label_id[0] == -1:
#         print(train_pos[i][0])
#         print(i+1)
#         print(train_sentences[i][0])
