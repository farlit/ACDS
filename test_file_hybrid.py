from metrics import start_of_chunk,end_of_chunk
def generate_file(sentences,seg_poses,write_file,flags=None):
    sentences = [sentence[1:] for sentence in sentences]    #sentence最前面有[CLS]符号
    with open(write_file, "a", encoding='utf-8-sig') as test:
        test.truncate(0)
    length = len(sentences)
    if flags is None:
        print(flags)
        flags = [0]*length
    for j,(sentence,seg_pos,flag) in enumerate(zip(sentences,seg_poses,flags)):
        prev_tag = 'O'
        begin_offset = 0
        seg = []
        pos = []
        for temp in seg_pos:
            seg.append(temp[0])
            pos.append(temp[1:])
        for i, chunk in enumerate(seg  + ['O']):
            tag = chunk[0]
            if end_of_chunk(prev_tag, tag):    # 判断当前字符(tag)的前一个字符(prev_tag)是不是结束字符
                with open(write_file, 'a', encoding='utf-8-sig') as f:
                    word = sentence[begin_offset:i]
                    pos_ = pos[begin_offset:i]
                    pos_.reverse()
                    f.write(''.join(word)+'/'+max(pos_,key=pos_.count)+' ')
            if start_of_chunk(prev_tag, tag):  # 判断当前字符是不是开始字符
                begin_offset = i
            prev_tag = tag
        if not flag or j==length-1 or not flags[j+1]:
            with open(write_file, 'a', encoding='utf-8-sig') as f:
                f.write('\n')
    f.close()