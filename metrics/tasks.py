# -*- coding: utf-8 -*-
import os

from abc import ABCMeta, abstractmethod

from .utils import load_tsv


class Task():
    '''Abstract class for a task.

    Methods and attributes:
        - load_data: load dataset from a path and create splits
        - yield dataset for training
        - dataset size
        - validate and test

    Outside the task:
        - process: pad and indexify data given a mapping
        - optimizer
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, n_classes):
        self.name = name
        self.n_classes = n_classes
        self.train_data_text = None
        self.val_data_text = None
        self.test_data_text = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_layer = None
        self.pair_input = 1
        self.categorical = 1
        self.val_metric = f'{self.name}\'s_accuracy'
        self.val_metric_decreases = False
        self.scorer1 = None
        self.scorer2 = None

    @abstractmethod
    def load_data(self, path, max_seq_len):
        '''
        Load data from path and create splits.
        '''
        raise NotImplementedError

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''
        acc = self.scorer1.get_metric(reset)
        return {'accuracy': acc}


class CoLATask(Task):
    '''Class for Warstdadt acceptability task'''

    def __init__(self, path, max_seq_len, name='acceptability'):
        ''' '''

        super(CoLATask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)
        self.val_metric_decreases = False
        self.scorer1 = None
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        '''Load the data'''

        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=None, targ_idx=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=3, s2_idx=None, targ_idx=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def get_metrics(self, reset=False):

        return {'accuracy': self.scorer1.get_metric(reset),
                'acc': self.scorer2.get_metric(reset)}


class SSTTask(Task):
    ''' Task class for Stanford Sentiment Treebank.  '''

    def __init__(self, path, max_seq_len, name="sst"):
        ''' '''

        super(SSTTask, self).__init__(name, 2)
        self.pair_input = 0
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' '''

        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len,
                            s1_idx=0, s2_idx=None, targ_idx=1, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=None, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data


class MRPCTask(Task):
    ''' Task class for Microsoft Research Paraphase Task.  '''

    def __init__(self, path, max_seq_len, name="mrpc"):
        ''' '''
        super(MRPCTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        ''' Process the dataset located at path.  '''

        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=0, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''

        acc = self.scorer1.get_metric(reset)
        prc, rcl, f1 = self.scorer2.get_metric(reset)
        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}


class STSBTask(Task):
    ''' Task class for Sentence Textual Similarity Benchmark.  '''

    def __init__(self, path, max_seq_len, name="sts_benchmark"):
        ''' '''
        super(STSBTask, self).__init__(name, 1)
        self.load_data(path, max_seq_len)
        self.categorical = 0
        self.val_metric_decreases = False
        self.scorer1 = None
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        ''' '''
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, skip_rows=1,
                           s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, skip_rows=1,
                            s1_idx=7, s2_idx=8, targ_idx=9, targ_fn=lambda x: float(x) / 5)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=7, s2_idx=8, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def get_metrics(self, reset=False):

        return {'accuracy': self.scorer1.get_metric(reset),
                'spearmanr': self.scorer2.get_metric(reset)}


class QQPTask(Task):
    '''
    Task class for Quora Question Pairs.
    '''

    def __init__(self, path, max_seq_len, name="quora"):
        ''' '''
        super(QQPTask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at data_file.'''

        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=3, s2_idx=4, targ_idx=5, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def get_metrics(self, reset=False):
        '''Get metrics specific to the task'''

        acc = self.scorer1.get_metric(reset)
        prc, rcl, f1 = self.scorer2.get_metric(reset)

        return {'accuracy': (acc + f1) / 2, 'acc': acc, 'f1': f1,
                'precision': prc, 'recall': rcl}


class MultiNLITask(Task):
    ''' Task class for Multi-Genre Natural Language Inference '''

    def __init__(self, path, max_seq_len, name="mnli"):
        '''MNLI'''

        super(MultiNLITask, self).__init__(name, 3)
        self.load_data(path, max_seq_len)
        self.scorer2 = None

    def load_data(self, path, max_seq_len):
        '''Process the dataset located at path.'''

        targ_map = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len,
                           s1_idx=8, s2_idx=9, targ_idx=11, targ_map=targ_map, skip_rows=1)
        val_matched_data = load_tsv(os.path.join(path, 'dev_matched.tsv'), max_seq_len,
                                    s1_idx=8, s2_idx=9, targ_idx=15, targ_map=targ_map, skip_rows=1)
        val_mismatched_data = load_tsv(os.path.join(path, 'dev_mismatched.tsv'), max_seq_len,
                                       s1_idx=8, s2_idx=9, targ_idx=15, targ_map=targ_map,
                                       skip_rows=1)
        val_data = [m + mm for m,
                    mm in zip(val_matched_data, val_mismatched_data)]
        val_data = tuple(val_data)

        te_matched_data = load_tsv(os.path.join(path, 'test_matched.tsv'), max_seq_len,
                                   s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_mismatched_data = load_tsv(os.path.join(path, 'test_mismatched.tsv'), max_seq_len,
                                      s1_idx=8, s2_idx=9, targ_idx=None, idx_idx=0, skip_rows=1)
        te_diagnostic_data = load_tsv(os.path.join(path, 'diagnostic.tsv'), max_seq_len,
                                      s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        te_data = [m + mm + d for m, mm, d in
                   zip(te_matched_data, te_mismatched_data, te_diagnostic_data)]
        te_data[3] = list(range(len(te_data[3])))

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data

    def get_metrics(self, reset=False):
        ''' No F1 '''

        return {'accuracy': self.scorer1.get_metric(reset)}


class QNLITask(Task):
    '''Task class for SQuAD NLI'''

    def __init__(self, path, max_seq_len, name="squad"):
        super(QNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''

        targ_map = {'not_entailment': 0, 'entailment': 1}
        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data


class RTETask(Task):
    ''' Task class for Recognizing Textual Entailment 1, 2, 3, 5 '''

    def __init__(self, path, max_seq_len, name="rte"):
        ''' '''

        super(RTETask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        ''' Process the datasets located at path. '''

        targ_map = {"not_entailment": 0, "entailment": 1}
        tr_data = load_tsv(os.path.join(path, 'train.tsv'), max_seq_len, targ_map=targ_map,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, 'dev.tsv'), max_seq_len, targ_map=targ_map,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)

        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data


class WNLITask(Task):
    '''Class for Winograd NLI task'''

    def __init__(self, path, max_seq_len, name="winograd"):
        ''' '''

        super(WNLITask, self).__init__(name, 2)
        self.load_data(path, max_seq_len)

    def load_data(self, path, max_seq_len):
        '''Load the data'''

        tr_data = load_tsv(os.path.join(path, "train.tsv"), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        val_data = load_tsv(os.path.join(path, "dev.tsv"), max_seq_len,
                            s1_idx=1, s2_idx=2, targ_idx=3, skip_rows=1)
        te_data = load_tsv(os.path.join(path, 'test.tsv'), max_seq_len,
                           s1_idx=1, s2_idx=2, targ_idx=None, idx_idx=0, skip_rows=1)
        self.train_data_text = tr_data
        self.val_data_text = val_data
        self.test_data_text = te_data
