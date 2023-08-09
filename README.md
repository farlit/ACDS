# Ancient-Chinese-Word-Segmentation-and-POS-Tagging-Using-Distant-Supervision

Code for ICASSP2023: Ancient Chinese Word Segmentation and Part-of-Speech Tagging Using Distant Supervision

we propose a novel augmentation method of ancient Chinese WSG and POS tagging data
using word alignment over parallel corpus

## Ⅰ、Get the labelled ancient Chinese and modern Chinese

 1、Get the raw unlabelled data:
 
 (1) Download the data from https://github.com/NiuTrans/Classical-Modern;
 
 (2) Rename the folder “双语数据” to “bitext” and place it under folder "ACDS";

 (3) Get modern Chinese sentences ("src") and ancient Chinese sentences ("tgt")
```
python make_data.py
```

 (4) Shuffle modern Chinese sentences and ancient Chinese sentences to get "src_shuf" and "tag_shuf".
```
python shuffle_data.py
```

 (5) Get the segmented、pos tagged modern Chinese data "src_shuf_seg、src_shuf_pos" by LTP (https://github.com/HIT-SCIR/ltp) and single- 
 character splited ancient Chinese data "tgt_shuf_seg"
```
python make_segpos_data.py 
``` 

## Ⅱ、Get the alignment probability between ancient Chinese and modern Chinese

 1、Download the alignment tool giza:
 
 (1) Download the tool giza from https://github.com/sillsdev/giza-py and put it in "ACDS" as a folder "giza";
 
 (2) Configure the tool giza as described in https://github.com/sillsdev/giza-py.

 2、Get the alignment probabilities 
 Get the alignment probabilities between segmented ancient words and single-character splited ancient Chinese words. The output result is file "alignment".
```
cd giza
python giza.py --source ../src_shuf_seg --target ../tgt_shuf_seg --alignments ../alignment --model ibm4 --m1 10 --mh 10 --m3 10 --m4 10 --include-probs
```

## Ⅲ、Get the word boundaries and parts of speech of ancient Chinese words

Get the labelled ancient Chinese data "tgt.shuf.seg_pos".
```
cd ..
python align-pos_tag_ltp.py
```

## Ⅳ、Train and evaluate the model

After getting the augmented data "tgt.shuf.seg_pos" from word alignment, you can train the SIKU-RoBerta (https://huggingface.co/SIKU-BERT/sikuroberta) like this repository (https://github.com/farlit/The-first-ancient-Chinese-word-segmentation-and-part-of-speech-tagging-code-and-analysis).

The augmented training dataset is "tgt.shuf.seg_pos"; the annotated training dataset is "zuozhuan_train_utf8"; the validation datasets are "EvaHan_testa_gold" and "EvaHan_testb_gold".

The annotated dataset and validation datasets are from https://github.com/RacheleSprugnoli/LT4HALA/tree/master/2022/data_and_doc. 

Our code refers to https://github.com/hemingkx/WordSeg, thanks for their work.
