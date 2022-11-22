from typing import Union

import torch

from .DeepTIMe import deeptime
from .DeepTIMe2 import deeptime2
from .DeepTIMe3 import deeptime3


def get_model(model_type: str, **kwargs: Union[int, float]) -> torch.nn.Module:
    if model_type == 'deeptime':
        model = deeptime(datetime_feats=kwargs['datetime_feats'], dim_size=kwargs['dim_size'])
    elif model_type=="deeptime2":
        model = deeptime2(datetime_feats=kwargs['datetime_feats'], dim_size=kwargs['dim_size'])
    elif model_type=="deeptime3":
        model = deeptime3(datetime_feats=kwargs['datetime_feats'], dim_size=kwargs['dim_size'])
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model
