# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .tasks import CoLATask, STS14Task, STSBTask


class SingleTaskModel(nn.Module):

    def __init__(self, args, encoder):
        super(SingleTaskModel, self).__init__()

        self.encoder = encoder
        self.dropout = args.dropout
        self.hidden_size = args.hid_dim

    def build_classifier(self, task, input_size):

        if isinstance(task, (STS14Task, STSBTask)):
            self.classifier = nn.Linear(input_size, task.n_classes)
        else:
            self.classifier = nn.Sequentail(
                nn.Dropout(p=self.dropout),
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.hidden_size, task.n_classes)
            )

    def forward(self, task, input1, input2=None, label=None):

        # TODO: if task.pair_input

        emb = self.encoder(input1)
        logits = self.classifier(emb)

        if label is not None:

            if isinstance(task, (STS14Task, STSBTask)):
                pass
            elif isinstance(task, CoLATask):
                pass
            else:
                label = label.squeeze(dim=-1)
                m = nn.CrossEntropyLoss()
                loss = m(logits, label)
                # TODO: README TODOs #2
                # task.scorer1(logits, label)
                # if task.scorer2 is not None:
                #     task.scorer2(logits, label)

        return logits, loss


class MultiTaskModel(nn.Module):
    pass


def build_model(args, tasks, bert):

    model = SingleTaskModel(args, bert)
    # TODO: Multi task
    # for task in tasks:
    model.build_classifier(tasks, args.emb_dim * 4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return model.to(device)

