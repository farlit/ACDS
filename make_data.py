import re
import os
import glob
from langconv import *

def tradition2simple(line):
    line = Converter('zh-hans').convert(line)
    return line

def simplified2traditional(sentence):
    sentence = Converter('zh-hant').convert(sentence)
    return sentence

file_name = 'bitext.txt'
root_folder = './bitext'
def Truncate(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.truncate()
            print(f"{file_path} has been truncated!")
    else:
        print(f"{file_path} doesn't exist!")
Truncate('src')
Truncate('tgt')
def merge_files(root_folder):
    for root, dirs, files in os.walk('./bitext'):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            data_raw = open(file_path, "r", encoding='utf-8').readlines()
            for i in range(0, len(data_raw), 3):
                ancient = data_raw[i][3:]
                modern = data_raw[i+1][4:]
                ancient = re.sub('[^\u4e00-\u9fa5。？！，、；：.?!,;:\'"’‘“”—…《》]', '', ancient)   #Remove exotic characters, including Spaces
                modern = re.sub('[^\u4e00-\u9fa5。？！，、；：.?!,;:\'"’‘“”—…《》]', '', modern)
                ancient = simplified2traditional(ancient)
                modern = tradition2simple(modern)
                if len(modern) - len(ancient) >= -1:
                    with open("src", 'a', encoding='utf-8') as f:
                        f.write(modern)
                        f.write('\n')
                    with open("tgt", 'a', encoding='utf-8') as f:
                        f.write(ancient)
                        f.write('\n')
            if dirs:
                for d in dirs:
                    merge_files(os.path.join(root, d))
merge_files('./bitext')
