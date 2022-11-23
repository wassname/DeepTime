# install environment

```sh
# try with pip torch WORKS!
export PROJ=deeptime
conda create -n $PROJ python=3.8 -y
conda activate $PROJ
mamba install -y ipykernel pip ipywidgets
pip install torch==1.10.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# 117 does not exist yet
python -m ipykernel install --user --name $PROJ
pip install gin-config fire pandas matplotlib numpy scikit-learn einops tensorboard yapf
pip install tsai

# note that I've also recorded the env in requirements

python -m experiments.forecast --config_path=storage/experiments/Exchange/192S/repeat=0/config.gin run | tee -a storage/experiments/Exchange/192S/repeat=0/instance.log 2>&1%
```

# run

```sh
python -m experiments.forecast --config_path=storage/experiments/Exchange/96S/repeat=0/config.gin run

python -m experiments.forecast --config_path=storage/experiments/Exchange/96Splus/repeat=0/config.gin run

python -m experiments.forecast --config_path=storage/experiments/Exchange/96Splusshort/repeat=0/config.gin run

python -m experiments.forecast --config_path=storage/experiments/Exchange/96Sshort/repeat=0/config.gin run
```


# Lessons

Single variate works much better. The output is not just a straight line. Likely because we have limited the output, not the input

# stocks

```sh
python -m experiments.forecast --config_path=experiments/configs/Stocks/96S.gin build_experiment 
python -m experiments.forecast --config_path=storage/experiments/Stocks/96S/repeat=0/config.gin run
```


```
make build-all path=experiments/configs/Stocks
./run.sh
```

# So how does deeptime work?

original:
- inr(coords)
- RR

My mods (I added past other variables):
- inr(concat([x, coords]))
- RR

Where INR is one of [mlp, lstm, lstm2, transformer, transforme2, inceptioncausal]

TODO:

- [x] try just one predictor
- [x] compare multi
- losses:
    - try logp? nah
    - mae?
- [x] make my own csv with 5m data (maybe 10k rows)
- [ ] backtest?

- [x] M2S mode
- [ ] add other INR's
- [ ] add None as lrn
- [ ] no enc?

```
python -m experiments.forecast --config_path=experiments/configs/hp_search/Stocks.gin build_experiment
./run.sh
```
