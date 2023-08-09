import random
with open('src', "r", encoding='utf-8') as f1, open('tgt', "r", encoding='utf-8') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

lines = list(zip(lines1, lines2))
random.seed(42)
random.shuffle(lines)

with open('src_shuf', 'w') as f1, open('tgt_shuf', 'w') as f2:
    for line1, line2 in lines:
        f1.write(line1)
        f2.write(line2)