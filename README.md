# Counter-Interference Adapter for Multilingual Machine Translation

### Overview

This is the (early-access) implementation of EMNLP'21 Finding paper [Counter-Interference Adapter for Multilingual Machine Translation](https://arxiv.org/abs/2104.08154)

The code is based on [NeurST Toolkits](https://github.com/bytedance/neurst.git)

#### Data prepare

Please follow `data/DataProcess.md` to download and preprocess data.

#### Training and Inference

The major class for the paper is TransformerCIAT in `neurst/models/transformer_CIAT.py`

The example config files are in `configs/`, please modify the file path in those yml files.

Pretaining:
```bash
python3 -m neurst.cli.run_exp.py --config configs/pretrain.yml
```

Tune on En-De with adapter:
```bash
python3 -m neurst.cli.run_exp.py --config configs/tune-on-en2de.yml
```
Inference:

```bash
python3 -m neurst.cli.run_exp.py --config configs/inference.yml
```
