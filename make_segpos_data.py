import os
import torch
from ltp import LTP
os.environ["CUDA_VISIBLE_DEVICES"]="6"
ltp = LTP('LTP/base1')
if torch.cuda.is_available():
    ltp.to('cuda')
data6=open('src_shuf', "r", encoding='utf-8').readlines()
data7=open('tgt_shuf', "r", encoding='utf-8').readlines()
def Truncate(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.truncate()
            print(f"{file_path} has been truncated!")
    else:
        print(f"{file_path} doesn't exist!")
Truncate('src_shuf_seg')
Truncate('src_shuf_pos')
Truncate('tgt_shuf_seg')
def add_to_tgt_seg(seg_ancients):
    with open("tgt_shuf_seg", 'a', encoding='utf-8') as f:
        for seg_ancient in seg_ancients:
            for seg in seg_ancient:
                f.write(seg)
                f.write(' ')
            f.write('\n')
def add_to_src_seg(seg_moderns):
    with open("src_shuf_seg", 'a', encoding='utf-8') as f:
        for seg_modern in seg_moderns:
            for seg in seg_modern:
                f.write(seg)
                f.write(' ')
            f.write('\n')
def add_to_src_pos(pos_moderns):
    with open("src_shuf_pos", 'a', encoding='utf-8') as f:
        for pos_modern in pos_moderns:
            for pos in pos_modern:
                f.write(pos)
                f.write(' ')
            f.write('\n')
seg_moderns = []
pos_moderns = []
seg_ancients = []
assert len(data6) == len(data7)
for i in range(len(data6)):
    modern = [data6[i][:-1]]      # :-1是为了去出 换行符号
    ancient = data7[i][:-1]

    seg_modern, pos_modern = ltp.pipeline(modern, tasks=['cws', 'pos']).to_tuple()
    seg_modern = seg_modern[0]
    pos_modern = pos_modern[0]
    seg_moderns.append(seg_modern)
    pos_moderns.append(pos_modern)

    seg_ancient = []
    for token in ancient:
        seg_ancient.append(token)
    seg_ancients.append(seg_ancient)

    if (i+1) % 1000==0:
        print(i+1)
        add_to_tgt_seg(seg_ancients)
        add_to_src_seg(seg_moderns)
        add_to_src_pos(pos_moderns)
        seg_moderns = []
        pos_moderns = []
        seg_ancients = []
add_to_tgt_seg(seg_ancients)
add_to_src_seg(seg_moderns)
add_to_src_pos(pos_moderns)
