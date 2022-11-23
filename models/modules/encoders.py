
# import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat, reduce

from tsai.models.InceptionTimePlus import (
    Conv,
    noop,
    nn,
    LinBnDrop,
    GAP1d,
    torch,
    AddCoords1d, BatchNorm
)
from tsai.models.TSTPlus import TSTPlus
from tsai.models.TSPerceiver import TSPerceiver
from tsai.models.TSSequencerPlus import TSSequencerPlus
from torch.nn.utils import weight_norm, spectral_norm
from .causalinception import CausalInceptionTimePlus
from .inr import INR

def custom_head(head_nf, c_out, seq_len):
    return nn.Sequential(
        # AddCoords1d(),
        # Conv(head_nf+1, head_nf, 2, bias=True, norm='Spectral'),
        # nn.BatchNorm1d(head_nf),
        # # nn.Dropout(0.15),
        # nn.ReLU(),
        AddCoords1d(),
        Conv(head_nf + 1, c_out, 1, bias=False, norm="Spectral"),
    )
    
class LinBnDropSN(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False, norm=None):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [spectral_norm(nn.Linear(n_in, n_out, bias=not bn))]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


class InceptionEncoder(nn.Module):
    def __init__(self, c_in, c_out, dropout, layers, layer_size, *args, **kwargs):
        super().__init__()
        self.net = CausalInceptionTimePlus(
            c_in=c_in, c_out=c_out, ks=[39, 19, 3], custom_head=custom_head, coord=True, fc_dropout=dropout, bn=True, depth=layers, nf=layer_size, *args, **kwargs
        )
        bn = kwargs.get("bn", True)
        fc_dropout = kwargs.get("fc_dropout", 0.15)
        self.pool = nn.Sequential(
            # GACP1d(1),
            # LinBnDrop(c_out*2, c_out, bn=bn, p=dropout)
            GAP1d(1),
            LinBnDropSN(c_out, c_out, bn=bn, p=fc_dropout),
        )
        self.head = nn.Sequential(
            # just to make sure we get a spectral norm final layer (after cat)
            LinBnDropSN(c_out*2, c_out, bn=bn, p=fc_dropout),
        )

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        outs = self.net(x.permute(0, 2, 1))  # .permute(0, 2, 1)
        last = outs[:, :, -1]  # take last
        max = self.pool(outs)
        return self.head(torch.cat([max, last], 1))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        layers=3,
        layer_size=512,
        dropout=0.1,
        n_heads=4,
        conv_dropout=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        # d_model (82) must be divisible by n_heads (4)
        layer_size = layer_size // n_heads * n_heads
        d_model = layer_size // 2
        self.net = TSTPlus(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_model // n_heads,
            d_v=d_model // n_heads,
            d_ff=layer_size,
            n_layers=layers,
            dropout=conv_dropout,
            fc_dropout=dropout,
            flatten=False,
            # *args, **kwargs
        )

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        outs = self.net(x.permute(0, 2, 1))
        return outs
    

class TransformerEncoder2(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        layers=3,
        layer_size=512,
        dropout=0.1,
        n_heads=4,
        conv_dropout=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        # d_model (82) must be divisible by n_heads (4)
        layer_size = layer_size // n_heads * n_heads
        d_model = layer_size // 2 
        self.net = TSPerceiver(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            
            # cat_szs=0, n_cont=0, 
            n_latents=layer_size, d_latent=layer_size//4, 
            # d_context=None, 
            self_per_cross_attn=1,
                #  share_weights=True, cross_n_heads=1, d_head=None, 
                 
                 
            # d_model=d_model,
            # d_k=d_model // n_heads,
            # d_v=d_model // n_heads,
            # d_ff=layer_size,
            
            self_n_heads=n_heads,
            n_layers=layers,
            attn_dropout=conv_dropout,
            fc_dropout=dropout,
        )

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        outs = self.net(x.permute(0, 2, 1))
        return outs


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        dropout=0.1,
        conv_dropout=0,
        layers=1,
        layer_size=100,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            c_in,
            layer_size,
            num_layers=layers,
            bias=True,
            batch_first=True,
            dropout=conv_dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout) if dropout else noop
        self.fc = nn.Linear(layer_size * (1 + 0), c_out)

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        # x = x.transpose(2,1)    # [batch_size x n_vars x seq_len] --> [batch_size x seq_len x n_vars]
        output, _ = self.rnn(
            x
        )  # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[
            :, -1
        ]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        return self.fc(self.dropout(output))



class LSTMEncoder2(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        seq_len,
        dropout=0.1,
        conv_dropout=0,
        layers=1,
        layer_size=100,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.rnn = TSSequencerPlus(
            c_in=c_in,
            c_out=c_out,
            seq_len=seq_len,
            d_model=layer_size,
            depth=layers,
            lstm_dropout=conv_dropout,
            fc_dropout=dropout,
            pre_norm=False, use_token=True, use_pe=True, 
            use_bn=False, 
        )

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        return self.rnn(
            x.transpose(2, 1)
        )


class MLPEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        scales=[0.01, 0.1, 1, 5, 10, 20, 50, 100],
        n_fourier_feats=4096,
        layers=2,
        layer_size=32,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.net = INR(
            in_feats=c_in,
            out_feats=layer_size,
            scales=scales,
            n_fourier_feats=n_fourier_feats,
            layers=layers,
            layer_size=layer_size,
        )
        self.head = nn.Linear(layer_size, c_out)

    def forward(self, x):
        """
        Takes in a sequence of shape (batch, sequence, features)
        and outputs a representation of shape (batch, features)
        """
        return self.head(self.net(x)[:, -1])
