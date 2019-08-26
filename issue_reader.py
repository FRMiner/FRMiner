# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     issue_reader
   Description :
   Author :       xmz
   date：          2019/7/9
-------------------------------------------------
"""
import json
import re
from typing import Dict, List
import logging

from allennlp.data import Field, Token
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
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


def replace_tokens(content):
    content = re.sub(r"\*\*I'm submitting a.+?\\r\\n\\r\\n\*\*", "", content)
    content = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', 'URL ', content)
    content = re.sub(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$', 'EMAIL ', content)
    content = re.sub(r'[0-9a-z]{10,}', 'HASH_ID ', content)
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


@DatasetReader.register("issue_reader")
class IssueReader(DatasetReader):
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
                 segment_sentences: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_stemmer=PorterStemmer())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._segment_sentences = segment_sentences or SpacySentenceSplitter()

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                if not line:
                    continue
                line = json.loads(line)
                report = split_issue_template(line['body'])
                report = self._segment_sentences.split_sentences(report)
                cmts = line['comments']
                # for rl in report:
                #     if len(self._tokenizer.tokenize(rl)) < 6:
                #         continue
                if len(report) == 0:
                    continue
                with open("check_report", "a") as f:
                    f.write(" ".join(report))
                    f.write("\n")
                comments = []
                for comment in cmts:
                    comment = replace_tokens(comment['body'])
                    # if len(self._tokenizer.tokenize(comment)) < 6:
                    #     continue
                    comments.append(comment)
                if len(comments) == 0:
                    continue
                labels = line['label']
                label = "feature request" if "feature" in labels else "other"
                yield self.text_to_instance(report, comments, label)

    @overrides
    def text_to_instance(self, report: List[str], comments: List[str], label: str) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        report = ListField([TextField([word for word in self._tokenizer.tokenize(line)],
                                      self._token_indexers)
                            for line in report])
        fields['report_tokens'] = report
        if len(comments) > 0:
            comments = ListField([TextField([word for word in self._tokenizer.tokenize(line)],
                                            self._token_indexers)
                                  for line in comments])
            fields['comments_tokens'] = comments
        if label is not None:
            fields['label'] = LabelField(label)
        # all_instance_fields_and_types: List[Dict[str, str]] = [{k: v.__class__.__name__
        #                                                         for k, v in Instance(fields).fields.items()}]
        # print(all_instance_fields_and_types)
        return Instance(fields)
