# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     issue_reader_siamese
   Description :
   date：          2019/7/28
-------------------------------------------------
"""
import json
import random
import re
from collections import defaultdict
from itertools import permutations
from typing import Dict, List
import logging

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Comment:
    user = None
    user_id = None
    user_type = None
    user_site_admin = None
    body = None


class Issue:
    number = None
    label = None
    owner = None
    title = None
    body = None
    comments = []


angular_template_split_pre = ["Please check if the PR fulfills these requirements"]
angular_template_split_post = ["What is the current behavior? (You can also link to an open issue here)"]


# TODO delete "morning hi hello hey"
def replace_tokens(content):
    content = re.sub(r"\*\*I'm submitting a.+?\\r\\n\\r\\n\*\*", "", content)
    content = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', 'URL ', content)
    content = re.sub(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$', 'EMAIL ', content)
    content = re.sub(r'[(0-9)+(a-z)+]{10,}', 'HASH_ID ', content)
    content = re.sub(r'#\d+\s', 'PR_ID ', content)
    content = re.sub(r"'''.*'''", "CODE ", content)
    content = re.sub(r'<[^>]*>|<\/[^>]*>', 'HTML ', content)
    content = re.sub(r'-\s\[\s*x?\s*\]\s((feature\srequest)|(bug\sreport)|other)', '', content)
    return content


def split_issue_template(content):
    for split_str in angular_template_split_pre:
        if split_str in content:
            content = content.split(split_str)[0]
            break
    for split_str in angular_template_split_post:
        if split_str in content:
            content = content.split(split_str)[1]
            break
    return replace_tokens(content)


@DatasetReader.register("dialog_reader")
class IssueReaderSiamese(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 segment_sentences: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True),
                                                     word_stemmer=PorterStemmer())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if segment_sentences:
            self._segment_sentences = SpacySentenceSplitter()
        self._class_cnt = defaultdict(int)

    def read_dataset(self, file_path):
        features = []
        others = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                if not line or len(line) == 0:
                    continue
                line = json.loads(line)
                if "id" not in line.keys():
                    d_id = ""
                else:
                    d_id = line['id']
                report = split_issue_template(line['body'])
                report = self._segment_sentences.split_sentences(report)
                cmts = line['comments']
                comments = []
                for comment in cmts:
                    user_name = comment['user']
                    comment = replace_tokens(comment['body'])
                    if len(comment) == 0:
                        continue
                    comments.append((user_name, comment))
                dialog = report + comments
                if len(dialog) == 0:
                    continue
                labels = line['label']
                if len(labels) == 0:
                    label = None
                else:
                    label = "feature" if "feature" in labels or "type: feature" in labels else "other"
                if "feature" == label:
                    features.append((d_id, dialog, label))
                else:
                    others.append((d_id, dialog, label))
        return features, others

    @overrides
    def _read(self, file_path):
        features, others = self.read_dataset(file_path)
        all_data = features + others
        random.shuffle(all_data)
        same_num = 0
        diff_num = 0
        if "unlabel" in file_path:
            logger.info("Begin predict------")
            features, others = self.read_dataset("frmodel/data/{}_target_train.txt")
            for sample in features + others:
                yield self.text_to_instance((sample, sample), is_gold=True)
            for sample in all_data:
                yield self.text_to_instance((sample, sample))
            logger.info(f"Predict sample num is {len(all_data)}")
        else:
            logger.info("Begin training-------")
            iter_num = 1
            if "test" in file_path:
                features, others = self.read_dataset(re.sub("test", "train", file_path))
                iter_num = 1
            for _ in range(iter_num):
                # plain balance data
                if "train" in file_path:
                    for k in range(len(others) - len(features)):
                        all_data.append(random.choice(features))
                for sample in all_data:
                    positive = random.choice(features)
                    negative = random.choice(others)
                    yield self.text_to_instance((sample, positive))
                    yield self.text_to_instance((sample, negative))
                    same_num += 1
                    diff_num += 1
            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")


    @overrides
    def text_to_instance(self, p, is_gold=False) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        ins1, ins2 = p
        dialog = ListField([TextField([word for word in self._tokenizer.tokenize(line[1])],
                                      self._token_indexers)
                            for line in ins1[1]])
        fields['dialog1'] = dialog
        fields["pos_tags1"] = ListField([SequenceLabelField([word.tag_ for word in self._tokenizer.tokenize(line[1])],
                                                            tokens, label_namespace="pos")
                                         for line, tokens in zip(ins1[1], dialog)])
        if ins1[-1] is not None and ins2[-1] is not None:
            if ins1[-1] == ins2[-1]:
                fields['label'] = LabelField("same")
            else:
                fields['label'] = LabelField("diff")
            fields['label_tags'] = LabelField("@".join([ins1[-1], ins2[-1]]), label_namespace="label_tags")
        fields['label'] = LabelField(ins1[-1])
        fields['metadata'] = MetadataField({"is_gold": is_gold, "pair_instance": p})

        return Instance(fields)
