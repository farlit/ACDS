def convert_tag(path):
    with open(path, encoding='utf-8-sig') as f:
        rd = [i for i in f.read().split('\n') if i]
    longlist = []
    indexlist = []
    row = 1
    for line in rd:
        for i in line.split(' '):
            if i:
                longlist.append(i)
                indexlist.append(row)
        row += 1
    output = []
    extra_word_num, extra_pos_num = 0, 0
    for chip, idx in zip(longlist, indexlist):
        sp = chip.split('/')
        sp_len = len(sp)
        if sp_len == 2:
            word = sp[0]
            pos = sp[1]
            if not pos:
                extra_word_num += 1
                pos = 'O'
                print('Line {} has words without pos tag.'.format(idx))
        else:
            if sp_len == 1:
                word = sp[0]
                extra_word_num += 1
                pos = 'O'
                print('Line {} has words without pos tag.'.format(idx))
            else:
                word = sp[0]
                pos = 'X'
                print('Line {} has redundant pos tags.'.format(idx))

        length = len(word)
        if length == 1:
            output.append([word, 'S-' + pos])
        elif length == 2:
            output.append([word[0], 'B-' + pos])
            output.append([word[-1], 'E-' + pos])
        else:
            if word:
                output.append([word[0], 'B-' + pos])
                for middle_c in word[1:-1]:
                    output.append([middle_c, 'I-' + pos])
                output.append([word[-1], 'E-' + pos])
            else: # lack character: !!!!!!!!!
                extra_pos_num += 1
                if pos != 'X':
                    print('Line {} has pos tags without word.'.format(idx))
    return output, extra_word_num, extra_pos_num


def count_prf(contestant, answer):
    outcome = dict()
    correct_wsg_num, correct_pos_num = 0, 0

    con_queue, con_ex_word, con_ex_pos = convert_tag(contestant)
    ans_queue, ans_ex_word, ans_ex_pos = convert_tag(answer)

    len_cq = len(con_queue)
    len_aq = len(ans_queue)
    origin, label = [], []
    for a_q in ans_queue:
        origin.append([a_q[0], a_q[-1].split('-')])
    for c_q in con_queue:
        label.append([c_q[0], c_q[-1].split('-')])

    o_index = 0
    l_index = 0
    jud = False
    while o_index < len_aq:
        now_org, now_lab = origin[o_index], label[l_index]
        noo_cha, nol_cha = now_org[0], now_lab[0]
        if noo_cha == nol_cha:
            noo_sp, nol_sp = now_org[1], now_lab[1]
            if noo_sp[0] == nol_sp[0] == 'S':
                correct_wsg_num += 1
                if noo_sp[1] == nol_sp[1]:
                    correct_pos_num += 1
                o_index += 1
                l_index += 1
            elif noo_sp[0] == nol_sp[0] == 'B':
                o_index += 1
                l_index += 1
                while (origin[o_index][0] == label[l_index][0]) and (origin[o_index][1][0] == label[l_index][1][0]):
                    if origin[o_index][1][0] == 'E':
                        correct_wsg_num += 1
                        if origin[o_index][1][1] == label[l_index][1][1]:
                            correct_pos_num += 1
                        o_index += 1
                        l_index += 1
                        break
                    o_index += 1
                    l_index += 1
                if origin[o_index][0] != label[l_index][0]:
                    while origin[o_index][0] != label[l_index][0]:
                        if len_cq > len_aq:
                            l_index += 1
                        elif len_cq < len_aq:
                            o_index += 1
            else:
                l_index += 1
                o_index += 1
        else:
            while origin[o_index][0] != label[l_index][0]:
                if len_cq > len_aq:
                    l_index += 1
                elif len_cq < len_aq:
                    o_index += 1

    machine_num, man_num = 0, 0

    for org in origin:
        bies = org[1][0]
        if bies in ['E', 'S']:
            man_num += 1
    for lab in label:
        bies = lab[1][0]
        if bies in ['E', 'S']:
            machine_num += 1


    pw = correct_wsg_num / machine_num
    rw = correct_wsg_num / man_num
    fw = (2 * pw * rw) / (pw + rw)
    outcome['WoS'] = [pw, rw, fw]
    pp = correct_pos_num / (machine_num + con_ex_pos - con_ex_word)
    rp = correct_pos_num / man_num
    fp = (2 * pp * rp) / (pp + rp)
    outcome['PoS'] = [pp, rp, fp]

    return outcome