# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dialog_model
   Description :
   date：          2019/7/9
-------------------------------------------------
"""

import logging
from typing import Dict, List, Any

from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, MultiHeadSelfAttention
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy, Auc, F1Measure, Metric, BooleanAccuracy, PearsonCorrelation, \
    Covariance
from allennlp.training.util import get_batch_size
from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity

from .siamese_metric import SiameseMeasure
from .util import pack2sequence

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def pad_sequence2len(tensor, dim, max_len) -> torch.LongTensor:
    shape = tensor.size()
    if shape[dim] >= max_len:
        return tensor
    pad_shape = list(shape)
    pad_shape[dim] = max_len - shape[dim]
    pad_tensor = torch.zeros(*pad_shape, device=tensor.device, dtype=tensor.dtype)
    new_tensor = torch.cat([tensor, pad_tensor], dim)
    return new_tensor


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance = CosineSimilarity()
        self.eps = 1e-6
        self.mse = torch.nn.MSELoss()

    def forward(self, output1, output2, target, size_average=True):
        distances = self.distance(output1, output2)
        losses = (1 - target.float()) * nn.functional.relu((self.margin - distances)).pow(2) \
                 + target.float() * (1 - distances).pow(2) / 4
        return losses.mean() if size_average else losses.sum(), distances


@Model.register("Dialog_Model")
class FRModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_tag_embedding: Embedding = None,
                 users_embedding: Embedding = None,
                 dropout: float = 0.1,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self._label_namespace = label_namespace
        self._dropout = Dropout(dropout)
        self._text_field_embedder = text_field_embedder
        self._pos_tag_embedding = pos_tag_embedding or None
        representation_dim = self._text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += self._pos_tag_embedding.get_output_dim()
        self._report_cnn = CnnEncoder(representation_dim, 25)
        self._comment_cnn = CnnEncoder(representation_dim, 25)
        lstm_input_dim = self._comment_cnn.get_output_dim()
        self._user_embedding = users_embedding or None
        if users_embedding is not None:
            lstm_input_dim += self._user_embedding.get_output_dim()
        rnn = nn.LSTM(input_size=lstm_input_dim,
                      hidden_size=150,
                      batch_first=True,
                      bidirectional=True)
        self._encoder = PytorchSeq2SeqWrapper(rnn)
        self._seq2vec = CnnEncoder(self._encoder.get_output_dim(), 25)
        self._num_class = self.vocab.get_vocab_size(self._label_namespace)
        self._bilinear_sim = BilinearSimilarity(self._encoder.get_output_dim(), self._encoder.get_output_dim())
        self._projector = FeedForward(self._seq2vec.get_output_dim(), 2,
                                      [50, self._num_class],
                                      Activation.by_name("sigmoid")(), dropout)
        self._golden_instances = None
        self._golden_instances_labels = None
        self._golden_instances_id = None
        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f-measure": F1Measure(positive_label=vocab.get_token_index("feature", "labels")),
        }
        self._loss = torch.nn.CrossEntropyLoss()
        self._contrastive_loss = ContrastiveLoss()
        self._mse_loss = torch.nn.MSELoss()
        initializer(self)

    @staticmethod
    def contrastive_loss(left, right):
        pairwise_distance = torch.nn.PairwiseDistance(p=1).cuda()
        return torch.exp(-pairwise_distance(left, right)).cuda()

    def _instance_forward(self,
                          dialog: Dict[str, torch.LongTensor],
                          users: torch.LongTensor = None,
                          pos_tags: torch.LongTensor = None):

        dialog['tokens'] = pad_sequence2len(dialog['tokens'], -1, 5)
        dialog_embedder = self._text_field_embedder(dialog)
        dialog_embedder = self._dropout(dialog_embedder)
        if pos_tags is not None and self._pos_tag_embedding is not None:
            pos_tags = pad_sequence2len(pos_tags, -1, 5)
            pos_tags_embedder = self._pos_tag_embedding(pos_tags)
            dialog_embedder = torch.cat([dialog_embedder, pos_tags_embedder], -1)
        dialog_mask = get_text_field_mask(dialog, num_wrapping_dims=1).float()
        dialog_shape = dialog_embedder.size()
        dialog_embedder = dialog_embedder.view(dialog_shape[0] * dialog_shape[1], -1,
                                               dialog_shape[-1])

        dialog_out = self._comment_cnn(dialog_embedder, dialog_mask.view(dialog_embedder.size()[:-1]))
        dialog_out = dialog_out.view(*dialog_shape[:2], -1)

        dialog_mask = torch.sum(dialog_mask, -1) > 0
        if users is not None and self._user_embedding is not None:
            users_embedder = self._user_embedding(users)
            dialog_out = torch.cat([users_embedder, dialog_out], -1)
        rnn_out = self._encoder(dialog_out, dialog_mask)
        rnn_out = pad_sequence2len(rnn_out, 1, 5)
        dialog_mask = pad_sequence2len(dialog_mask, -1, 5)
        rnn2vec = self._seq2vec(rnn_out, dialog_mask)
        return rnn2vec

    def forward_gold_instances(self, d_id, dialog, user, pos_tag, label_tags):
        if self._golden_instances is None:
            self._golden_instances = torch.tensor(self._instance_forward(dialog, user, pos_tag))
        else:
            self._golden_instances = torch.cat([self._golden_instances, self._instance_forward(dialog, user, pos_tag)])
        if self._golden_instances_labels is None:
            self._golden_instances_labels = torch.tensor(label_tags)
        else:
            self._golden_instances_labels = torch.cat([self._golden_instances_labels, label_tags])
        if self._golden_instances_id is None:
            self._golden_instances_id = [d_id]
        else:
            self._golden_instances_id.append(d_id)

    def forward(self,
                dialog1: Dict[str, torch.LongTensor],
                dialog2: Dict[str, torch.LongTensor] = None,
                users1: torch.LongTensor = None,
                pos_tags1: torch.LongTensor = None,
                users2: torch.LongTensor = None,
                pos_tags2: torch.LongTensor = None,
                label: torch.IntTensor = None,
                label_tags: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        output_dict = dict()
        if metadata is not None and not metadata[0]["is_gold"]:
            output_dict["pair_instance"] = [meta["pair_instance"] for meta in metadata]
        if metadata is not None and metadata[0]["is_gold"]:
            self.forward_gold_instances(metadata[0]["pair_instance"][0][0], dialog1, users1, pos_tags1, label_tags)
            return output_dict
        rnn_vec1 = self._instance_forward(dialog1, users1, pos_tags1)
        if self._golden_instances is not None:
            logits = []
            for gold in self._golden_instances:
                logits.append(self._projector(torch.cat([rnn_vec1, gold.unsqueeze(0)], -1)))
            output_dict['logits'] = logits

        else:
            logits = self._projector(rnn_vec1)
            probs = nn.functional.softmax(logits, dim=-1)
            output_dict["logits"] = logits
            output_dict["probs"] = probs
            if label is not None:
                loss = self._loss(logits, label)
                output_dict['loss'] = loss
                output_dict['label_tags'] = label_tags
                for metric_name, metric in self._metrics.items():
                    metric(logits, label)
        return output_dict

    def inference(self, predictions, label_tags, d_id):
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = (self.vocab.get_index_to_token_vocabulary(self._label_namespace)
                         .get(label_idx, str(label_idx)))
            classes.append(label_str)
        golden_names = []
        if len(label_tags) == 1:
            label_tags = label_tags.expand(len(classes))
        for tags in label_tags:
            if tags == self.vocab.get_token_index("feature@feature", "label_tags"):
                golden_names.append("feature")
            if tags == self.vocab.get_token_index("other@other", "label_tags"):
                golden_names.append("other")
        predict_labels = []
        pos_ins = []
        neg_ins = []
        for class_name, golden_name in zip(classes, golden_names):
            if class_name == "same":
                if golden_name == "feature":
                    predict_labels.append(1 / 31)
                    pos_ins.append(d_id)
                elif golden_name == "other":
                    predict_labels.append(0)
            elif class_name == "diff":
                if golden_name == "feature":
                    predict_labels.append(0)
                elif golden_name == "other":
                    predict_labels.append(1 / 261)
        # TODO return gold_id
        return pos_ins + neg_ins, predict_labels

    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        if 'logits' not in output_dict.keys():
            output_dict["label"] = ["gold"]
            return output_dict
        if isinstance(output_dict['logits'], list):
            infered = np.array(
                [self.inference(nn.functional.softmax(logits, dim=-1), label_tags, d_id) for logits, label_tags, d_id in
                 zip(output_dict['logits'], self._golden_instances_labels.unsqueeze(1), self._golden_instances_id)])
            vote_id = infered[:, 0]
            vote_id = [v[0] for v in vote_id if len(v) > 0]
            predict_socres = infered[:, 1]
            predict_socres = [float(p[0]) for p in predict_socres]
            predict_socres = np.sum(predict_socres, -1)
            output_dict["vote"] = []
            ins_id = []
            is_feature = []
            if len(vote_id) > 10:
                ins_id = vote_id
            ins_id = sorted(ins_id)
            if len(ins_id) > 0:
                is_feature.append(" || ".join(ins_id))
                output_dict["vote"].append(len(ins_id))
            else:
                is_feature.append("")
                output_dict["vote"].append(0)
        else:
            _, _ = self.inference(output_dict['probs'], output_dict) > 0

        predict_labels = ["feature" if len(label) > 0 else "other" for label in is_feature]
        output_dict["label"] = predict_labels
        for pred, ins, vote, vote_ins in zip(predict_labels, output_dict["pair_instance"], output_dict["vote"],
                                             is_feature):
            if "feature" in pred:
                dialog_id = ins[0][0]
                with open("bs_pred_fr.txt", "a", encoding="utf8") as f:
                    f.write(f"ID: {dialog_id}\tVote: {vote}\tVote_Ins: {vote_ins}\n")

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        metrics['accuracy'] = self._metrics['accuracy'].get_metric(reset)
        precision, recall, fscore = self._metrics['f-measure'].get_metric(reset)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['fscore'] = fscore
        return metrics
