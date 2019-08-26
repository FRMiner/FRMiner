# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       xmz
   date：          2019/7/9
-------------------------------------------------
"""
import json
import unittest

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.nn.util import batched_index_select

from .issue_reader import IssueReader


class ModelTest(unittest.TestCase):
    def json_test(self):
        with open("../IRC/angular_issue3.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            print(json.loads(line))

    def reader_test(self):
        reader = IssueReader()
        instances = reader.read("data/train.txt")
        # fields = instances[0].fields
        # print([t.text for tt in fields["report_tokens"] for t in tt.tokens])
        # print([t.text for t in fields["comments_tokens"].tokens])
        # print(fields["label"].label)
        return instances

    def iterator_test(self):
        instances = self.reader_test()
        vocab = Vocabulary.from_files("out/vocabulary")
        iterator = BucketIterator(sorting_keys=[["report_tokens1", "num_fields"]], batch_size=3)
        iterator.index_with(vocab)
        generator = iterator(instances=instances, num_epochs=None)
        batch = next(generator)
        print(batch)

    def masked_scatter_test(self):
        a = torch.tensor([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0], [1, 0, 0, 0, 0]])
        mask = torch.ByteTensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
        b = torch.tensor([5])
        print(a)
        print(torch.masked_select(a, mask))

    def batched_index_select_test(self):
        # indices = numpy.array([[[1, 2],
        #                         [3, 4]],
        #                        [[5, 6],
        #                         [7, 8]]])
        # # Each element is a vector of it's index.
        # targets = torch.ones([2, 10, 4]).cumsum(1) - 1
        # # Make the second batch double it's index so they're different.
        # targets[1, :, :] *= 2
        # print(targets)
        # indices = torch.tensor(indices, dtype=torch.long)
        # selected = batched_index_select(targets, indices)
        # print(selected)
        a = torch.tensor([[1, 2, 3, 0, 0], [2, 3, 4, 5, 0], [1, 0, 0, 0, 0]])
        indices = [[1, 2, 3], [1, 2], [1]]
        indices = torch.LongTensor(indices)
        print(batched_index_select(a, indices))

    def zip_test(self):
        a = torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6,
                          6, 6, 6, 7, 7, 8, 8, 9])
        b = torch.tensor([1, 8, 15, 10, 2, 4, 4, 19, 6, 16, 4, 2, 18, 1, 3, 1, 16, 2,
                          5, 8, 8, 13, 6, 13, 6, 14, 7, 30, 2, 13, 5, 30])
        for k in enumerate(zip(a, b)):
            print(k)


if __name__ == '__main__':
    test = ModelTest()
    test.zip_test()
