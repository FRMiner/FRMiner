# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predictor
   Description :
   Author :       xmz
   date：          2019/7/13
-------------------------------------------------
"""
import json
import random

from allennlp.models import Model
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.predictors.predictor import Predictor

from .issue_reader import split_issue_template, replace_tokens


@Predictor.register('fr_predictor')
class FRPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model

    allennlp predict out/siamese_bs_no_trans/ FRMiner/data/bootstrap_unlabel.txt --output-file out.txt --cuda-device 0 --use-dataset-reader --include-package=FRMiner
    """



    # def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
    #     super().__init__(model, dataset_reader)
    #     self._golden_features, self._golden_others = dataset_reader.read_dataset("FRMiner/data/angular_target_all.txt")
    #
    # def predict(self, sample: str) -> JsonDict:
    #     sample = json.loads(sample)
    #     return self.predict_batch_instance(self._json_to_instance(sample))
    #
    # def parse_dialog(self, line):
    #     report = split_issue_template(line['body'])
    #     report = self._segment_sentences.split_sentences(report)
    #     cmts = line['comments']
    #     comments = []
    #     for comment in cmts:
    #         comment = replace_tokens(comment['body'])
    #         comments.append(comment)
    #     dialog = report + comments
    #     labels = line['label']
    #     label = "feature" if "feature" in labels or "type: feature" in labels else "other"
    #     return dialog, label
    #
    # @overrides
    # def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    #     dialog1, label1 = self.parse_dialog(json_dict)
    #     positive = random.choice(self._golden_features)
    #     negative = random.choice(self._golden_others)
    #     return [self._dataset_reader.text_to_instance(((dialog1, label1), positive)),
    #             self._dataset_reader.text_to_instance(((dialog1, label1), negative))]



