# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(self, query_reprs:Tensor, context_reprs: Tensor, context_y: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(context_reprs, context_y, reg_coeff)
        preds = self.forecast(query_reprs, w, b)
        return preds
    
    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.transpose(-2, -1), X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.transpose(-2, -1), Y)
            weights = torch.linalg.solve(A, B)
        else:
            # Woodbury
            # A = torch.bmm(X, X.mT)
            A = torch.bmm(X, X.transpose(-2, -1))
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            # weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))
            weights = torch.bmm(X.transpose(-2, -1), torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)
