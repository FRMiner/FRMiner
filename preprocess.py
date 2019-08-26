# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_preprocess
   Description :
   Author :       xmz
   date：          2019/7/13
-------------------------------------------------
"""
import json
import random


def data_split(source_fr, source_other, fold_num):
    with open(source_fr, 'r') as f:
        frs = f.readlines()
    with open(source_other, 'r') as f:
        others = f.readlines()
    all_data = frs + others
    print(len(all_data))
    fr = []
    other = []
    for d in all_data:
        d = d.strip()
        if not d or len(d) == 0:
            continue
        js = json.loads(d)
        if js['body'] is not None:
            if js['label'] == "feature":
                fr.append(js)
            elif js['label'] == 'other':
                other.append(js)
    class_statistic(fr + other)
    random.shuffle(fr)
    random.shuffle(other)
    fr_fold_num = len(fr) // fold_num
    other_fold_num = len(other) // fold_num
    pos_folds = []
    neg_folds = []
    for i in range(fold_num):
        if i == fold_num - 1:
            pos_folds.append(fr[i * fr_fold_num:])
            neg_folds.append(other[i * other_fold_num:])
        else:
            pos_folds.append(fr[i * fr_fold_num:(i + 1) * fr_fold_num])
            neg_folds.append(other[i * other_fold_num:(i + 1) * other_fold_num])
    train_folds = []
    test_folds = []
    for i in range(fold_num):
        train = []
        test = []
        for j in range(fold_num):
            if j == i:
                test.extend(neg_folds[j])
                test.extend(pos_folds[j])
            else:
                train.extend(pos_folds[j])
                train.extend(neg_folds[j])
        train_folds.append(train)
        test_folds.append(test)

    return train_folds, test_folds


def class_statistic(data):
    fr_cnt = 0
    other_cnt = 0
    for d in data:
        labels = d['label']
        if "feature" in labels:
            fr_cnt += 1
        else:
            other_cnt += 1
    print(f"Feature Request: {fr_cnt}, Others: {other_cnt}, Rate: {fr_cnt / (other_cnt + fr_cnt + 1e-6)}")


def make_data():
    project = "chromium_target"
    fold_num = 3
    train_folds, test_folds = data_split(f"data/{project}_feature.txt", f"data/{project}_other.txt", fold_num)
    for i in range(fold_num):
        train = train_folds[i]
        test = test_folds[i]
        print(f"Train statistic at fold {i}")
        class_statistic(train)
        print(f"Test statistic at fold {i}")
        class_statistic(test)
        random.shuffle(train)
        random.shuffle(test)
        with open(f"data/{project}_train_{i}.txt", "a") as f:
            for line in train:
                f.write(json.dumps(line) + "\n")
        with open(f"data/{project}_test_{i}.txt", "a") as f:
            for line in test:
                f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    make_data()
