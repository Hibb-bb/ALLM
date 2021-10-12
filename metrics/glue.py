# -*- coding: utf-8 -*-

from .tasks import (CoLATask, MRPCTask, MultiNLITask, QNLITask, QQPTask,
                    RTETask, SSTTask, STSBTask, WNLITask)

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


def build_tasks(args):
    pass

def convert2id(seq, vocab):
    pass

def padding(seq, max_len):
    pass