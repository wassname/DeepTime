
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.regressors import RidgeRegressor
from models.modules.inr import INR, INRLayer

class SumHead(nn.Module):
    def __init__(self, d, c_out=1, dropout=0):
        super().__init__()
        # self.conv = nn.Sequential(
        #     CausalConv1d(head_nf, c_out, 1, bias=False, norm="Spectral"),
        # )
        self.l = nn.Sequential(
            INRLayer(d, d, dropout=dropout),
            # INRLayer(d, d, dropout=dropout),
            nn.Linear(d, c_out)
        ) # nn.Linear(d, c_out) # init a random transform

    def forward(self, query, support, support_labels):
        return self.l(query)


class TransformerHead(nn.Module):
    def __init__(self, d, c_out=1, dropout=0.1, num_heads=16):
        super().__init__()
        if d<64:
            num_heads = 4
        d = d//num_heads*num_heads # make sure it's divisable
        hidden_dim = d//4
        # the value is just one class, so let's embed it first
        self.value_encoder = nn.Sequential(
            INRLayer(c_out, hidden_dim, dropout=0),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.l = nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True, kdim=d, vdim=hidden_dim, add_bias_kv=True, bias=True, dropout=0)
        # after using attention let's decode it
        self.decoder = nn.Sequential(
            INRLayer(d, d, dropout=dropout),
            nn.Linear(d, c_out)
        )

    def forward(self, query, support, support_labels, *args, **kwargs):
        """
        Fits the support set with ridge regression and 
        returns the classification score on the query set.

        Parameters:
            query:  a (tasks_per_batch, n_query, d) Tensor.
            support:  a (tasks_per_batch, n_support, d) Tensor.
            support_labels: a (tasks_per_batch, n_support) Tensor.
            n_way: a scalar. Represents the number of classes in a few-shot classification task.
            n_shot: a scalar. Represents the number of support examples given per class.
            lambda_reg: a scalar. Represents the strength of L2 regularization.
        Returns: a (tasks_per_batch, n_query, n_way) Tensor.
        """
        # should be (batch, seq, feature)
        value = self.value_encoder(support_labels)
        attn_output, _ = self.l(query=query, key=support, value=value)
        o = self.decoder(attn_output)
        return o


class RegressionHead(nn.Module):
    def __init__(self, base_learner='Ridge', d=512, enable_scale=True, dropout=0.1, num_heads=16):
        super().__init__()
        if ('Ridge' in base_learner):
            # the regular DeepTime one
            self.head = RidgeRegressor()
        elif ("None" in base_learner):
            self.head = SumHead(d=d, dropout=dropout)
        elif ("Transformer" in base_learner):
            self.head = TransformerHead(d=d, dropout=dropout, num_heads=num_heads)
        else:
            raise NotImplementedError(base_learner)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, query, support, support_labels, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, **kwargs)
        else:
            return self.head(query, support, support_labels, **kwargs)
