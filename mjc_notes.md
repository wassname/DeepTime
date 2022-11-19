```sh
# try with pip torch WORKS!
export PROJ=deeptime
conda create -n $PROJ python=3.8 -y
conda activate $PROJ
mamba install -y ipykernel pip ipywidgets
pip install torch==1.10.0+cu113 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# 117 does not exist yet
python -m ipykernel install --user --name $PROJ
pip install gin-config fire pandas matplotlib numpy scikit-learn einops tensorboard


python -m experiments.forecast --config_path=storage/experiments/Exchange/192S/repeat=0/config.gin run >> storage/experiments/Exchange/192S/repeat=0/instance.log 2>&1%
```
