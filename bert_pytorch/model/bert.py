import torch.nn as nn

from .transformer import TransformerBlock, TransformerBlock_AL
from .embedding import BERTEmbedding, BERTEmbedding_AL


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class BERTAL(nn.Module):
    """
    BERT AL model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, config=None):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.detach = config["detach"]
        self.g_dim = config["g_dim"]
        self.h_dim = config["h_dim"]
        self.act = config["act"]

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding_AL(vocab_size=vocab_size, embed_size=hidden)

        # TODO: switch to ALBlock, and remove ModuleList if it doesn't work.
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock_AL(hidden, attn_heads, hidden * 4, dropout, self.g_dim, self.h_dim, self.act, self.detach) for _ in range(n_layers)])

    def forward(self, x, segment_info, y):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)

        loss_data = {}
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x, y, loss = self.embedding(x, segment_info, y)
        
        loss_data["emb associated loss"] = self.embedding.al
        loss_data["emb AE loss"] = self.embedding.ael

        # running over multiple transformer blocks
        for idx, transformer in enumerate(self.transformer_blocks):
            x, y, l = transformer.forward(x, mask)
            loss += l
            loss_data[f"transformer layer{idx} associated loss"] = transformer.ass_loss
            loss_data[f"transformer layer{idx} AE loss"] = transformer.ae_loss

        return loss, loss_data

    def inference(self, x, y):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding.inference(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.inference(x, mask)
        y = self.transformer_blocks[-1].b(x)
        for transformer in reversed(self.transformer_blocks):
            y = transformer.h(y)
        y_pred = self.embedding.h(y)

        return y_pred

    def encode(self, x, mode='tran'):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding.inference(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.inference(x, mask)

        y = self.transformer_blocks[-1].b(x)
        for transformer in reversed(self.transformer_blocks):
            y = transformer.h(y)
        # y_pred = self.embedding.h(y)
        if mode == 'tran':
            return x # (batch, seq_len, hidden)
        if mode == 'small':
            return y #(batch, seq_len, h_dim)