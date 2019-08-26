# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     siamese_metric
   Description :
   Author :       xmz
   date：          2019/7/31
-------------------------------------------------
   Change Activity:
                   2019/7/31:
-------------------------------------------------
"""
from typing import Optional

import torch
from allennlp.training.metrics import Metric


@Metric.register("siamese_measure")
class SiameseMeasure(Metric):

    def __init__(self, vocab) -> None:
        self._correct_cnt = []
        self._total_cnt = []
        self._voacb = vocab

        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

    def __call__(self,
                 predictions: torch.Tensor,
                 pair_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        predictions, pair_labels, mask = self.unwrap_to_tensors(predictions, pair_labels, mask)
        num_classes = predictions.size(-1)
        predictions = predictions.view((-1, num_classes))
        predictions = predictions.max(-1)[1].unsqueeze(-1)

        same_label = self._voacb.get_token_index("same", "labels")
        diff_label = self._voacb.get_token_index("diff", "labels")
        ff_label = self._voacb.get_token_index("feature@feature", "label_tags")
        fo_label = self._voacb.get_token_index("feature@other", "label_tags")
        of_label = self._voacb.get_token_index("other@feature", "label_tags")
        oo_label = self._voacb.get_token_index("other@other", "label_tags")

        for p, pair_label in zip(predictions, pair_labels):
            if same_label == p:
                if ff_label == pair_label:
                    self.true_positive += 1
                if oo_label == pair_label:
                    self.true_negative += 1
                if fo_label == pair_label:
                    self.false_positive += 1
                if of_label == pair_label:
                    self.false_negative += 1
            if diff_label == p:
                if fo_label == pair_label:
                    self.true_negative += 1
                if of_label == pair_label:
                    self.true_positive += 1
                if ff_label == pair_label:
                    self.false_negative += 1
                if oo_label == pair_label:
                    self.false_positive += 1

    def get_metric(self, reset: bool):
        precision = self.true_positive * 1.0 / (self.true_positive + self.false_positive + 1e-6)
        recall = self.true_positive * 1.0 / (self.true_positive + self.false_negative + 1e-6)
        fmeasure = (2.0 * precision * recall) / (precision + recall + 1e-6)
        return precision, recall, fmeasure

    def reset(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0
