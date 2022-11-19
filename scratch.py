# %%

import os
from os.path import join
import math
import logging
from typing import Callable, Optional, Union, Dict, Tuple

from matplotlib import pyplot as plt

import gin
from fire import Fire
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from experiments.base import Experiment
from data.datasets import ForecastDataset
from models import get_model
from utils.checkpoint import Checkpoint
from utils.ops import default_device, to_tensor
from utils.losses import get_loss_fn
from utils.metrics import calc_metrics

from experiments.forecast import get_data
gin.enter_interactive_mode()
# %%


gin.clear_config()
# gin.parse_config(open("storage/experiments/Exchange/96M/repeat=0/config.gin"))
gin.parse_config(open("storage/experiments/Exchange/96Mplus/repeat=0/config.gin"))

# %%
train_set, train_loader = get_data(flag='train', batch_size=16)
# x, _, _, _ =train_set[0]
# x = x * 1.0

# %%
# x -= x[0]
# x /= x.std()
# plt.plot(x)
# # %%


# %%
model = get_model("deeptime2",
                    dim_size=train_set.data_x.shape[1],
                    datetime_feats=train_set.timestamps.shape[-1]).to(default_device())
model.load_state_dict(torch.load('storage/experiments/Exchange/96Mplus/repeat=0/model.pth'))
model = model.eval()

# %%
b = train_set[1]
b = [bb[None, :] for bb in b]
x, y, x_time, y_time = map(to_tensor, b)
with torch.no_grad():
    forecast = model(x, x_time, y_time)
# %%


# %%
plt.title('inception inr')
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.keys())
l = x.shape[1]
forecast2 = forecast[0].detach().cpu().numpy()
x2 = x[0].cpu()
y2 = y[0].cpu()
i_past = list(range(l))
i_future = list(range(l, l*2))
for i in range(x.shape[-1]):
    plt.plot(range(l), x2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(range(l, l*2), y2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(range(l, l*2), forecast2[:, i], c=colors[i], linestyle='--')

# %%
gin.clear_config()
gin.parse_config(open("storage/experiments/Exchange/96M/repeat=0/config.gin"))

train_set, train_loader = get_data(flag='train', batch_size=16)

model = get_model("deeptime",
                    dim_size=train_set.data_x.shape[1],
                    datetime_feats=train_set.timestamps.shape[-1]).to(default_device())
model.load_state_dict(torch.load('storage/experiments/Exchange/96M/repeat=0/model.pth'))
model = model.eval()


b = train_set[1]
b = [bb[None, :] for bb in b]
x, y, x_time, y_time = map(to_tensor, b)
with torch.no_grad():
    forecast = model(x, x_time, y_time)


plt.title('mlp inr')
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.keys())
l = x.shape[1]
forecast2 = forecast[0].detach().cpu().numpy()
x2 = x[0].cpu()
y2 = y[0].cpu()
i_past = list(range(l))
i_future = list(range(l, l*2))
for i in range(x.shape[-1]):
    plt.plot(range(l), x2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(range(l, l*2), y2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(range(l, l*2), forecast2[:, i], c=colors[i], linestyle='--')
# %%


gin.clear_config()
gin.parse_config(open("storage/experiments/Exchange/96Mplus2/repeat=0/config.gin"))

train_set, train_loader = get_data(flag='train', batch_size=16)

model = get_model("deeptime2",
                    dim_size=train_set.data_x.shape[1],
                    datetime_feats=train_set.timestamps.shape[-1]).to(default_device())
model.load_state_dict(torch.load('storage/experiments/Exchange/96Mplus2/repeat=0/model.pth'))
model = model.eval()


b = train_set[1]
b = [bb[None, :] for bb in b]
x, y, x_time, y_time = map(to_tensor, b)
with torch.no_grad():
    forecast = model(x, x_time, y_time)


plt.title('inception inr')
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.keys())
l = x.shape[1]
forecast2 = forecast[0].detach().cpu().numpy()
x2 = x[0].cpu()
y2 = y[0].cpu()
l2 = y.shape[1]
i_past = list(range(l))
i_future = list(range(l, l+l2))
for i in range(x.shape[-1]):
    plt.plot(i_past, x2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(i_future, y2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(i_future, forecast2[:, i], c=colors[i], linestyle='--')

# %%


gin.clear_config()
gin.parse_config(open("storage/experiments/Exchange/96M2/repeat=0/config.gin"))

train_set, train_loader = get_data(flag='train', batch_size=16)

model = get_model("deeptime",
                    dim_size=train_set.data_x.shape[1],
                    datetime_feats=train_set.timestamps.shape[-1]).to(default_device())
model.load_state_dict(torch.load('storage/experiments/Exchange/96M2/repeat=0/model.pth'))
model = model.eval()


b = train_set[1]
b = [bb[None, :] for bb in b]
x, y, x_time, y_time = map(to_tensor, b)
with torch.no_grad():
    forecast = model(x, x_time, y_time)


plt.title('mlp inr2')
import matplotlib.colors as mcolors
colors = list(mcolors.BASE_COLORS.keys())
l = x.shape[1]
forecast2 = forecast[0].detach().cpu().numpy()
x2 = x[0].cpu()
y2 = y[0].cpu()
l2 = y.shape[1]
i_past = list(range(l))
i_future = list(range(l, l+l2))
for i in range(x.shape[-1]):
    plt.plot(i_past, x2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(i_future, y2[:, i], c=colors[i])
for i in range(x.shape[-1]):
    plt.plot(i_future, forecast2[:, i], c=colors[i], linestyle='--')

# %%
