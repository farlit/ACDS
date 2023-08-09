all_data=open("zuozhuan_train_noise.txt", 'r', encoding='utf-8').readlines()
with open("zuozhuan_train_noise0.txt", 'a', encoding='utf-8') as f:
    for data in all_data:
        if data == '\n':
            continue
        head_data = data.split()[0]
        head_pos  = head_data.split('/')[1]
        while head_pos == 'null':
            data = data[len(head_data)+1:]
            if not data:
                break
            head_data = data.split()[0]
            head_pos = head_data.split('/')[1]
        if data:
            f.write(data)