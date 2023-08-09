import os
import torch
import label
import config
import numpy as np
from tqdm import tqdm
from test_file import generate_file
from metrics import f1_score_seg,f1_score_pos
def evaluate(dev_loader, model1, model2, model3, model4, mode='eval'):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
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
            batch_output1 = model1(batch_data, token_type_ids=None, attention_mask=batch_masks, \
                                       seglabels=batch_seglabels, poslabels=batch_poslabels)
            batch_segoutput1, batch_posoutput1 = batch_output1[0]

            batch_output2 = model2(batch_data, token_type_ids=None, attention_mask=batch_masks, \
                                   seglabels=batch_seglabels, poslabels=batch_poslabels)
            batch_segoutput2, batch_posoutput2 = batch_output2[0]

            batch_output3 = model3(batch_data, token_type_ids=None, attention_mask=batch_masks, \
                                   seglabels=batch_seglabels, poslabels=batch_poslabels)
            batch_segoutput3, batch_posoutput3 = batch_output3[0]

            batch_output4 = model4(batch_data, token_type_ids=None, attention_mask=batch_masks, \
                                   seglabels=batch_seglabels, poslabels=batch_poslabels)
            batch_segoutput4, batch_posoutput4 = batch_output4[0]

            batch_segoutput = (batch_segoutput1 + batch_segoutput2 + batch_segoutput3 + batch_segoutput4) / 4
            batch_posoutput = (batch_posoutput1 + batch_posoutput2 + batch_posoutput3 + batch_posoutput4) / 4

            labelseg_masks = batch_masks[:,1:]  # get padding mask
            labelpos_masks = batch_masks[:,1:]

            batch_segoutput = batch_segoutput.detach().cpu().numpy()
            batch_posoutput = batch_posoutput.detach().cpu().numpy()

            for i, indices in enumerate(np.argmax(batch_segoutput, axis=2)):
                pred_tag = []
                for j, idx in enumerate(indices):
                    if labelseg_masks[i][j]:
                        pred_tag.append(id_seg2label.get(idx))
                pred_segtags.append(pred_tag)

            for i, indices in enumerate(np.argmax(batch_posoutput, axis=2)):
                pred_tag = []
                for j, idx in enumerate(indices):
                    if labelpos_masks[i][j]:
                        pred_tag.append(id_pos2label.get(idx))
                pred_postags.append(pred_tag)

            # batch_segoutput = model.crf_seg.decode(batch_segoutput, mask=batch_masks[:,1:])
            # pred_segtags.extend([[id_seg2label.get(idx) for idx in indices] for indices in batch_segoutput])
            #
            # batch_posoutput = model.crf_pos.decode(batch_posoutput, mask=batch_masks[:,1:])
            # pred_postags.extend([[id_pos2label.get(idx) for idx in indices] for indices in batch_posoutput])

            torch.cuda.empty_cache()
    return pred_segtags,pred_postags

