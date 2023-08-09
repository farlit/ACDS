import re
data_modern_seg=open('src.shuf.seg', "r", encoding='utf-8').readlines()
data_modern_pos=open('src.shuf.pos', "r", encoding='utf-8').readlines()
data_ancient_seg=open('tgt.shuf.seg', "r", encoding='utf-8').readlines()
align_pairs=open("alignment", 'r', encoding='utf-8').readlines()
assert len(data_modern_seg)==len(data_modern_pos)
assert len(data_modern_seg)==len(data_modern_pos)
assert len(data_modern_seg)==len(align_pairs)
# 'null' indicates no correspondence
convert_tags={'a':'a',   'b':'a',    'c':'c',    'd':'d',   'e':'y',   'h':'null',\
              'i':'null','j':'null', 'k':'null', 'm':'m',   'n':'n',   'nd':'f',\
              'nh':'nr', 'ni':'ns',  'nl':'n',   'ns':'ns', 'nt':'t',  'nz':'n',\
              'o':'s',   'p':'p',    'q':'q',    'r':'r',   'u':'u',   'v':'v',\
              'wp':'w',  'ws':'null','x':'null', 'g':'null','z':'a'}
for i in range(len(align_pairs)):
    modern_seg = data_modern_seg[i][:-2].split()
    modern_pos = data_modern_pos[i][:-2].split()
    ancient_sen = data_ancient_seg[i][:-2].split()
    align_pair = align_pairs[i][:-2].split()
    length_pair = len(align_pair)
    align_dict= [[(-1, 0, 'null')] for i in range(len(ancient_sen))]
    # (x,y,z) indicates modern word index, alignment confidence, the part of speech of aligned ancient Chinese word
    for j in range(length_pair):
        index_confidence_align = align_pair[j].split(':')
        index_align = index_confidence_align[0].split('-')
        confidence_align = float(index_confidence_align[1])
        index_modern = int(index_align[0])
        index_ancient = int(index_align[1])
        if index_modern >= len(modern_pos):
            break
        pos_ancient = convert_tags[modern_pos[index_modern]]
        if pos_ancient != 'null':
            if confidence_align>align_dict[index_ancient][-1][1]:
                align_dict[index_ancient].append((index_modern, confidence_align, pos_ancient))
    labelled_data=''
    for j in range(len(ancient_sen)):
        labelled_data += ancient_sen[j]
        if j < len(ancient_sen) - 1:
            if align_dict[j][-1][0] != align_dict[j+1][-1][0]:
                labelled_data += ('/'+align_dict[j][-1][-1]+' ')
        else:
            labelled_data += ('/'+align_dict[j][-1][-1]+'\n')
            break
    with open("tgt.shuf.seg_pos", 'a', encoding='utf-8') as f:
        if not labelled_data:
            labelled_data = '\n'
        f.write(labelled_data)
