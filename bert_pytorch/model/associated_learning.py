# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import dropout


class ALComponent(nn.Module):

    x: Tensor
    y: Tensor
    loss_b: Tensor
    loss_d: Tensor
    _s: Tensor
    _t: Tensor
    _s_prime: Tensor
    _t_prime: Tensor

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        bx: nn.Module,
        dy: nn.Module,
        cb: nn.Module,
        ca: nn.Module,
        dropout: float = 0.1
    ) -> None:

        super(ALComponent, self).__init__()

        self.f = f
        self.g = g
        # birdge function
        self.bx = bx
        # h function
        self.dy = dy
        # loss function
        self.criterion_br = cb
        self.criterion_ae = ca

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x)
            return self._s.detach(), self._t_prime.detach()

    def loss(self):

        self.loss_b = self.criterion_br(self.bx(self._s), self._t)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d


class TransformerEncoderAL(ALComponent):
    '''x: encoder, y: linear'''

    def __init__(
        self,
        d_model: Tuple[int, int],
        nhead: int,
        y_hidden: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        act: nn.Module = None,
    ) -> None:

        if act == None:
            act = nn.ELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model[0], nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first
        )
        # num layer = 1
        f = nn.TransformerEncoder(encoder_layer, 1)
        g = nn.Sequential(
            nn.Linear(d_model[1], y_hidden, bias=False),
            act
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(d_model[0], y_hidden, bias=False),
            act
        )
        # h function
        dy = nn.Sequential(
            nn.Linear(y_hidden, d_model[1], bias=False),
            act
        )
        # loss function
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

        super().__init__(f, g, bx, dy, cb, ca)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            self._s_prime = self.bx(self._s)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)

            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            output = self.dy(y)

            return self._s.detach(), output.detach()

    def loss(self):

        p = self._s_prime
        q = self._t

        # mean
        p_nonzero = (p != 0.).sum(dim=1)
        p = p.sum(dim=1) / p_nonzero

        self.loss_b = self.criterion_br(p, q)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d

    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        Shape: (sz, sz).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

"""
class TransformerEncoderAL(ALComponent):
    '''x: encoder, y: encoder'''

    def __init__(
        self,
        d_model: Tuple[int, int],
        nhead: int,
        y_hidden: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        act: nn.Module = None,
    ) -> None:

        if act == None:
            act = nn.ELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model[0], nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first
        )
        # num layer = 1
        f = nn.TransformerEncoder(encoder_layer, 1)
        g = nn.Sequential(
            nn.Linear(d_model[1], y_hidden, bias=False),
            act
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(d_model[0], y_hidden, bias=False),
            act
        )
        # h function
        dy = nn.Sequential(
            nn.Linear(y_hidden, d_model[1], bias=False),
            act
        )
        # loss function
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

        super().__init__(f, g, bx, dy, cb, ca)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            self._s_prime = self.bx(self._s)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)

            return self._s.detach(), self._t.detach()

        else:

            self._s = self.f(x, src_mask, src_key_padding_mask)
            output = self.dy(y)

            return self._s.detach(), output.detach()

    def loss(self):

        p = self._s_prime
        q = self._t

        # mean
        p_nonzero = (p != 0.).sum(dim=1)
        p = p.sum(dim=1) / p_nonzero

        self.loss_b = self.criterion_br(p, q)
        self.loss_d = self.criterion_ae(self._t_prime, self.y)

        return self.loss_b + self.loss_d

    def _generate_square_subsequent_mask(self, sz: int):
        '''
        Generate a square mask for the sequence. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        Shape: (sz, sz).
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask
"""