import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
import torch

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerBlock_AL(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, g_dim, h_dim, act=nn.Identity(), detach=True):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param g_dim: label embedding dimension
        :param h_dim: label input embedding
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.g = nn.Sequential(
            nn.Linear(h_dim, g_dim),
            act
        )
        self.h = nn.Sequential(
            nn.Linear(g_dim, h_dim),
            act
        )
        self.b = nn.Sequential(
            nn.Linear(hidden, g_dim),
            act
        )
        self.cri = nn.MSELoss()
        self.detach = detach

    def forward(self, x, mask, y, y_mask):

        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.dropout(self.output_sublayer(x, self.feed_forward))
        y_emb = self.g(y.float())
        y_back = self.h(y_emb)
        # y_mask = (y > 0).float().unsqueeze(1)
        x_b = self.b(torch.bmm(y_mask, x).squeeze(1))

        """
        compute loss
        """
        ass_loss = self.cri(y_emb, x_b)
        ae_loss = self.cri(y_emb, y_back)
        loss = ass_loss + ae_loss
        self.ass_loss = ass_loss.item()
        self.ae_loss = ae_loss.item()

        if self.detach:
            x = x.detach()
            y_emb = y_emb.detach()

        return x, y_emb, loss

    def inference(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
