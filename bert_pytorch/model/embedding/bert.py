import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
import torch

class LabelSmooth(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=0):
        super(LabelSmooth, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        # logs = self.log_softmax(logits)
        logs = logits
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


class BERTEmbedding_AL(nn.Module):
    """
    BERT Embedding AL which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, g_dim, act=nn.Identity(), dropout=0.1, detach=True):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        class_num = vocab_size
        
        self.g = nn.Sequential(
            nn.Linear(embed_size, g_dim),
            act
        )
        
        self.h = nn.Sequential(
            nn.Linear(g_dim, class_num),
            nn.LogSoftmax(dim=-1)
        )

        self.embed_size = embed_size
        self.ass_loss = nn.MSELoss()
        self.ae_loss = LabelSmooth()
        self.detach = detach

    def forward(self, sequence, segment_label, y):
        x = self.dropout(self.token(sequence) + self.position(sequence))#  + self.segment(segment_label)
        # print('\nx', x.shape)
        # print('y', y.shape)

        y_emb = self.dropout(self.g(self.token(y) + self.position(y)))
        y_pred = self.h(y_emb)
        # print('y emb',y_emb.shape)
        y_mask = (y > 0).float().unsqueeze(1)
        # print('y mask', y_mask.shape)
        y_emb = torch.bmm(y_mask, y_emb).squeeze(1)
        x_bridge = self.g(torch.bmm(y_mask, x).squeeze(1))

        # print('y emb', y_emb.shape) 
        # print('x bridge', x_bridge.shape)
        ass_loss = self.ass_loss(x_bridge, y_emb)
        # y_pred = self.dropout(self.h(y_emb))
        # print('y pred', y_pred.shape)
        ae_loss = self.ae_loss(y_pred.transpose(1,2), y)
        self.al = ass_loss.item()
        self.ael = ae_loss.item()
        # print('al', self.al)
        # print('ael', self.ael)
        # print(y_mask)
        if self.detach:
            x = x.detach()
            y_emb = y.detach()

        # print('x',x.shape, 'y',y_emb.shape)
        return x, y_emb, ae_loss + ass_loss, y_mask

    def inference(self, x):
        return self.token(x) + self.position(x)
