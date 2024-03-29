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
    id_segpos2label = label.id_segpos2label

    true_segtags = []
    pred_segtags = []

    true_postags = []
    pred_postags = []

    true_tags = []
    pred_tags = []

    sent_data = []
    flags = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            batch_data, batchseg_labels, batchpos_labels, sentences, flag = batch_samples
            sent_data.extend(sentences)
            flags.extend(flag)
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
            # batch_output = batch_output[0]

            labelseg_masks = batch_seglabels.gt(-1)  # get padding mask
            labelpos_masks = batch_poslabels.gt(-1)
            # labelsegpos_masks = batch_poslabels.gt(-1)

            batch_segoutput = model.crf_seg.decode(batch_segoutput, mask=labelseg_masks)
            batch_seglabels = batch_seglabels.to('cpu').numpy()
            pred_segtags.extend([[id_seg2label.get(idx) for idx in indices] for indices in batch_segoutput])
            # true_segtags.extend([[id_seg2label.get(idx) for idx in indices if idx > -1] for indices in batch_seglabels])

            batch_posoutput = model.crf_pos.decode(batch_posoutput, mask=labelpos_masks)
            batch_poslabels = batch_poslabels.to('cpu').numpy()
            pred_postags.extend([[id_pos2label.get(idx) for idx in indices] for indices in batch_posoutput])
            # batch_output = model.crf_segpos.decode(batch_output, mask=labelsegpos_masks)
            # pred_tags.extend([[id_segpos2label.get(idx) for idx in indices] for indices in batch_output])

            # true_postags.extend([[id_pos2label.get(idx) for idx in indices if idx > -1] for indices in batch_poslabels])
            # batch_segoutput = batch_segoutput.detach().cpu().numpy()
            # batch_posoutput = batch_posoutput.detach().cpu().numpy()
            #
            #
            # for i, indices in enumerate(np.argmax(batch_segoutput, axis=2)):
            #     pred_tag = []
            #     for j, idx in enumerate(indices):
            #         if labelseg_masks[i][j]:
            #             pred_tag.append(id_seg2label.get(idx))
            #     pred_segtags.append(pred_tag)
            #
            # for i, indices in enumerate(np.argmax(batch_posoutput, axis=2)):
            #     pred_tag = []
            #     for j, idx in enumerate(indices):
            #         if labelpos_masks[i][j]:
            #             pred_tag.append(id_pos2label.get(idx))
            #     pred_postags.append(pred_tag)

    if mode=='test':
        # os.remove(r'testb_舟几口_1_closed.txt')
        generate_file(sent_data, pred_segtags, pred_postags, 'testa_noise.txt', flags)
        return
    # assert len(pred_segtags) == len(true_segtags)
    # assert len(sent_data) == len(true_segtags)
    #
    # assert len(pred_postags) == len(true_postags)
    # assert len(sent_data) == len(true_postags)
    # logging loss, f1 and report
    seg_metrics = {}
    pos_metrics = {}
    f1, p, r = f1_score_seg(true_segtags, pred_segtags)
    seg_metrics['f1'] = f1
    seg_metrics['p'] = p
    seg_metrics['r'] = r
    # if mode != 'dev':
    #     bad_case(sent_data, pred_tags, true_tags)
    #     output_write(sent_data, pred_tags)
    #     output2res()
    seg_metrics['loss'] = dev_losses / len(dev_loader)

    f1_, p_, r_ = f1_score_pos(true_segtags, pred_segtags)
    pos_metrics['f1'] = f1_
    pos_metrics['p'] = p_
    pos_metrics['r'] = r_
    return seg_metrics,pos_metrics