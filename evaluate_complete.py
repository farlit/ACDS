import os
import torch
import label
import config
import numpy as np
from tqdm import tqdm
from test_file import generate_file
from metrics import f1_score_seg,f1_score_pos
def evaluate(dev_loader, model, mode='eval'):
    model.eval()
    id_seg2label = label.id_seg2label
    id_pos2label = label.id_pos2label
    pred_segtags = []
    pred_postags = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            batch_data, batchseg_labels, batchpos_labels, batchsegpos_labels, \
            batchgram_list, matching_matrix, channel_ids, _, _0 = batch_samples
            # shift tensors to GPU if available
            batch_data = batch_data.to(config.device)
            batch_seglabels = batchseg_labels.to(config.device)
            batch_poslabels = batchpos_labels.to(config.device)
            batch_masks = batch_data.gt(0).to(config.device)  # get padding mask
            # compute model output and loss
            # shape: (batch_size, max_len, num_labels)
            batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks, \
                                       seglabels=batch_seglabels, poslabels=batch_poslabels)

            batch_segoutput, batch_posoutput = batch_output[0]

            batch_segoutput = model.crf_seg.decode(batch_segoutput, mask=batch_masks[:,1:])
            pred_segtags.extend([[id_seg2label.get(idx) for idx in indices] for indices in batch_segoutput])

            batch_posoutput = model.crf_pos.decode(batch_posoutput, mask=batch_masks[:,1:])
            pred_postags.extend([[id_pos2label.get(idx) for idx in indices] for indices in batch_posoutput])

            torch.cuda.empty_cache()
    return pred_segtags,pred_postags

