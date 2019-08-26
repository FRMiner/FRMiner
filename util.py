# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     util
   Description :
   Author :       xmz
   date：          2019/7/12
-------------------------------------------------
"""
import torch
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_mask_from_sequence_lengths, sort_batch_by_length


def pack2sequence(seq1, mask1, seq2, mask2):
    seq1_lens = get_lengths_from_binary_sequence_mask(mask1)
    seq2_lens = get_lengths_from_binary_sequence_mask(mask2)
    combined_lens = seq1_lens + seq2_lens
    max_len, _ = torch.max(combined_lens + torch.tensor([5], device=seq1.device), dim=0)
    combined_tensor = torch.zeros(combined_lens.size()[-1], max_len, seq1.size()[-1], device=seq1.device)
    # print(combined_tensor.size())
    # print(combined_lens)
    for i, (len1, len2) in enumerate(zip(seq1_lens, seq2_lens)):
        combined_tensor[i, :len1, :] = seq1[i, :len1, :]
        combined_tensor[i, len1:len1 + len2, :] = seq2[i, :len2, :]
    combined_mask = get_mask_from_sequence_lengths(combined_lens, max_len)
    sorted_tensor, _, restoration_indices, permutation_index = sort_batch_by_length(
        combined_tensor, combined_lens)
    combined_mask = combined_mask.index_select(0, permutation_index)
    return sorted_tensor, combined_mask, restoration_indices, permutation_index
