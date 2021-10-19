# -*- coding: utf-8 -*-

import os
import pickle as pkl
from typing import List

from .tasks import (CoLATask, MRPCTask, MultiNLITask, QNLITask, QQPTask,
                    RTETask, SSTTask, STSBTask, WNLITask)
from .utils import convert2id, multi_class_process

# custom path
PATH_PREFIX = '../data/'

# {'task name': (load data, data path)}
NAME2INFO = {
    'cola': (CoLATask, 'CoLA/'),
    'sst': (SSTTask, 'SST-2/'),
    'mrpc': (MRPCTask, 'MRPC/'),
    'sts-b': (STSBTask, 'STS-B/'),
    'qqp': (QQPTask, 'QQP'),
    'mnli': (MultiNLITask, 'MNLI/'),
    'qnli': (QNLITask, 'QNLI/'),
    'rte': (RTETask, 'RTE/'),
    'wnli': (WNLITask, 'WNLI/')
}

# add prefix to each task
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

# -*- trainer -*-

def main(args):
    pass

# -*- task field -*-

def build_tasks(args):

    tasks = get_tasks(args.tasks, args.max_seq_len, args.load_tasks)
    vocab = get_vocab(args.vocab)

    for task in tasks:

        train, val, test = process_task(task, vocab)
        task.train_data = train
        task.val_data = val
        task.test_data = test

    return tasks


def get_tasks(task_names: List, max_seq_len, load):
    '''
    Load tasks.

    Args:
        * task_names: (single task) ['cola'] ; (multi task) task list.
        * max_seq_len
        * load: flag. 1 if task pickle exist.
    '''
    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        pkl_path = NAME2INFO[name][1] + f'{name}_task.pkl'
        if os.path.isfile(pkl_path) and load:
            task = pkl.load(open(pkl_path, 'rb'))
        else:
            task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        tasks.append(task)

    return tasks


def get_vocab(path):

    vocab = {}

    with open(path, 'r', encoding='utf8') as f:
        data = f.readlines()

    for i, word in enumerate(data):
        vocab[word] = i

    return vocab


def process_task(task, vocab):

    if hasattr(task, 'train_data_text') and task.train_data_text is not None:
        train = process_split(task.train_data_text, vocab,
                              task.pair_input, task.categorical, task.n_classes)
    else:
        train = None

    if hasattr(task, 'val_data_text') and task.val_data_text is not None:
        val = process_split(task.val_data_text, vocab,
                            task.pair_input, task.categorical, task.n_classes)
    else:
        val = None

    if hasattr(task, 'test_data_text') and task.test_data_text is not None:
        test = process_split(task.test_data_text, vocab,
                             task.pair_input, task.categorical, task.n_classes)
    else:
        test = None

    return train, val, test


def process_split(split, vocab, pair_input, categorical, nclass):

    if pair_input:

        pass

    else:
        text = split[0]
        # text:
        # [['_input', 'example'], ['_subword']]
        inputs = [convert2id(sent, vocab) for sent in text]
        # padding

        if categorical:
            labels = multi_class_process(split[2], nclass)
        else:
            labels = split[2]

        if len(split) == 4:
            # TODO: multi_class_process
            idxs = split[3]
            return inputs, labels, idxs
        else:
            return inputs, labels
