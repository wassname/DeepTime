 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat, reduce

from models.modules.metareghead import RegressionHead
from models.modules.causalinception import CausalInceptionTimePlus
from models.modules.inrplus2 import INRPlus2
from models.modules.inr import INR
from models.modules.encoders import LSTMEncoder, TransformerEncoder2, TransformerEncoder, InceptionEncoder, LSTMEncoder2, MLPEncoder

# from models.modules.regressors import RidgeRegressor

@gin.configurable()
def deeptime3(dim_size:int, datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float, dropout: float, lrn: str, enc:str, inr: str, seq_len: int):
    return DeepTIMe3(dim_size, datetime_feats, layer_size, inr_layers, n_fourier_feats, scales, dropout, lrn, enc, inr, seq_len)


class DeepTIMe3(nn.Module):
    def __init__(self, dim_size: int, datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float, dropout: float=0.3, lrn:str='Ridge', enc:str='inception', inr:str='INR', seq_len: int=46):
        super().__init__()
        
        # encode the past
        encoded_size = layer_size
        encoder_features = 24
        encoder_layers = 3
        if enc == 'inception':
            self.enc = InceptionEncoder(
                c_in=dim_size, c_out=encoded_size, dilation=6,
                layer_size=17, layers=encoder_layers, dropout=dropout,
            )
        elif enc == 'lstm':
            self.enc = LSTMEncoder(c_in=dim_size, c_out=encoded_size, dropout=dropout, layers=encoder_layers, layer_size=24)
        elif enc == 'lstm2':
            self.enc = LSTMEncoder2(c_in=dim_size, c_out=encoded_size, dropout=dropout, layers=encoder_layers, layer_size=32, seq_len=seq_len)
        elif enc == 'mlp':
            self.enc = MLPEncoder(c_in=dim_size, c_out=encoded_size, dropout=dropout, layers=encoder_layers, layer_size=256)
        elif enc == 'transformer':
            self.enc = TransformerEncoder(c_in=dim_size, c_out=encoded_size, dropout=dropout, layers=encoder_layers, layer_size=256, seq_len=seq_len)
        elif enc == 'transformer2':
            self.enc = TransformerEncoder2(c_in=dim_size, c_out=encoded_size, dropout=dropout, layers=encoder_layers, layer_size=128, seq_len=seq_len)
        elif enc == 'none':
            self.enc = None
            encoded_size = 0
        else:
            raise NotADirectoryError(enc)
        
        # translate coords to a representation, given a summary of the past
        coord_size = 1
        in_feats=datetime_feats+encoded_size+coord_size
        if inr=='INRPlus2':
            self.inr = INRPlus2(in_feats=in_feats, out_feats=layer_size, layers=inr_layers, layer_size=max(17, layer_size//8),
                       n_fourier_feats=n_fourier_feats//4, scales=scales, dropout=dropout)
        elif inr=="INR":
            self.inr = INR(in_feats=in_feats, out_feats=layer_size, layers=inr_layers, layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats, scales=scales, dropout=dropout)
        else:
            raise NotImplementedError(inr)
        
        # meta learn y given a representation
        self.regressionhead = RegressionHead(lrn=lrn, d=layer_size, dropout=dropout)

        self.datetime_feats = datetime_feats
        self.inr_layers = inr_layers
        self.layer_size = layer_size
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        
    def encode_and_decode(self, past_x, time, offset=0):
        """
        h_past = encode(past) # get representation of past
        representation = decode(h_past, coords)
        i = length of past, so we can offset the coords
        """
        
        # we summarize the past into a single hidden layer. Then repeat it for each coordinate
        past_len = time.shape[1]
        if self.enc is not None:
            encoded_x = self.enc(past_x)
            encoded_x = repeat(encoded_x, "b f -> b t f", t=past_len)
        

        # relative coordinates are the same for each batch, so we make them once and repeat them   
        coords = self.get_coords(past_len).to(time.device) + offset
        coords = repeat(coords, "1 t 1 -> b t 1", b=time.shape[0])
        
        # combine and run INR to decode the representation
        if self.enc is not None:
            context_input = torch.cat([encoded_x, coords, time], dim=-1)
        else:
            context_input = torch.cat([coords, time], dim=-1)
        context_repr = self.inr(context_input)
        return context_repr

    def forward(self, context_past_x, context_y, query_past_x, query_y, context_time, query_time) -> Tensor:

        context_reprs = self.encode_and_decode(context_past_x, context_time)
        query_reprs = self.encode_and_decode(query_past_x, query_time, offset=context_reprs.shape[1])

        preds = self.regressionhead(query_reprs, context_reprs, context_y)
        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b
    
    def get_coords(self, lookback_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len)
        return rearrange(coords, 't -> 1 t 1')
