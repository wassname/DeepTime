# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.modules.feature_transforms import GaussianFourierFeatureTransform

from tsai.models.InceptionTimePlus import InceptionTimePlus
from .causalinception import CausalInceptionTimePlus, CausalConv1d

def custom_head(head_nf, c_out, seq_len):
    return nn.Sequential(
        CausalConv1d(head_nf, c_out, 1, bias=False)
        
    )

class INRPlus2(nn.Module):
    def __init__(self, in_feats: int, layers: int, layer_size: int, n_fourier_feats: int, scales: float,
                 dropout: Optional[float] = 0.5, bn=False, *args, **kwargs):
        super().__init__()
        self.features = nn.Linear(in_feats, layer_size) if n_fourier_feats == 0 \
            else GaussianFourierFeatureTransform(in_feats, n_fourier_feats, scales)
        in_size = layer_size if n_fourier_feats == 0 \
            else n_fourier_feats+in_feats
        # import pdb; pdb.set_trace()
        self.layers = CausalInceptionTimePlus(
            in_size-1, layer_size, seq_len=None, nf=layer_size, depth=layers,
                 flatten=False, concat_pool=False, fc_dropout=dropout, conv_dropout=0.05, bn=bn, y_range=None, custom_head=custom_head, ks=[139, 19, 3], dilation=2, *args, **kwargs
        )
        # layers = [INRPlusLayer(in_size, layer_size, dropout=dropout)] + \
        #          [INRPlusLayer(layer_size, layer_size, dropout=dropout) for _ in range(layers - 1)]
        # self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # import pdb; pdb.set_trace()
        return self.layers(x.permute((0, 2, 1))).permute((0, 2, 1))
