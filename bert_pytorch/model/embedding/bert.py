import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


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

    def __init__(self, vocab_size, embed_size, class_num, g_dim, act=nn.Identity(), dropout=0.1):
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
        self.g = nn.Sequential(
            nn.Embedding(class_num, g_dim, padding=0),
            act
        )
        self.h = nn.Sequential(
            nn.Linear(g_dim, class_num),
            nn.LogSoftmax(dim=-1)
        )

        self.b = nn.Sequential(
            nn.Linear(embed_size, g_dim),
            act
        )
        self.embed_size = embed_size
        self.ass_loss = nn.MSELoss()
        self.ae_loss = nn.NLLLoss()

    def forward(self, sequence, segment_label, y):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        y_emb = self.dropout(self.g(y))
        x_bridge = self.dropout(self.b(x))
        ass_loss = self.ass_loss(x_bridge, y_emb)
        y_pred = self.dropout(self.h(y_emb))
        ae_loss = self.ae_loss(y_pred, y)
        self.al = ass_loss.item()
        self.ael = ae_loss.item()

        return self.dropout(x), ae_loss + ass_loss
