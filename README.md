<p align="center">
  <img src="http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/neurst_newlogo.png" height="200">
</p>

![Last Commit](https://img.shields.io/github/last-commit/bytedance/neurst)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.6%7C3.7%7C3.8-brightgreen)](https://github.com/bytedance/neurst)
[![Contributors](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](to_be_add)

The primary motivation of NeurST is to facilitate NLP researchers to get started on end-to-end speech translation (ST) and build advanced neural machine translation (NMT) models. 

**See [here](/examples) for a full list of NeurST examples. And we present recent progress of end-to-end ST technology at [https://st-benchmark.github.io/](https://st-benchmark.github.io/).** 

> NeurST is based on TensorFlow2 and we are working on the pytorch version.

## <img src="http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/icon-for-new-9.jpg" width="45">NeurST News
**March 29, 2022**: Release of [GigaST dataset](/datasets/GigaST): a large-scale speech translation corpus.

**Aug 16, 2021**: Release of models and results for [IWSLT 2021 offline ST and simultaneous translation task](/examples/iwslt21).

**June 15, 2021**: Integration of [LightSeq](https://github.com/bytedance/lightseq) for training speedup, see the [experimental branch](https://github.com/bytedance/neurst/tree/lightseq).

**March 28, 2021**: The v0.1.1 release includes the instructions of weight pruning and quantization aware training for transformer models, and several more features. See the [release note](https://github.com/bytedance/neurst/releases/tag/v0.1.1) for more details.

**Dec. 25, 2020**: The v0.1.0 release includes the overall design of the code structure and recipes for training end-to-end ST models. See the [release note](https://github.com/bytedance/neurst/releases/tag/v0.1.0) for more details.


## Highlights

- **Production ready**: The model trained by NeurST can be directly exported as TF savedmodel format and use TensorFlow-serving. There is no gap between the research model and production model. Additionally, one can use [LightSeq](https://github.com/bytedance/lightseq) for NeurST model serving with a much lower latency.
- **Light weight**: NeurST is designed specifically for end-to-end ST and NMT models, with clean and simple code. It has no dependency on Kaldi, which simplifies installation and usage.
- **Extensibility and scalability**: NeurST has the careful design for extensibility and scalability. It allows users to customize `Model`, `Task`, `Dataset` etc. and combine each other.
- **High computation efficiency**: NeurST has high computation efficiency and can be further optimized by enabling mixed-precision and XLA. Fast distributed training using [`Byteps`](https://github.com/bytedance/byteps) / [`Horovod`](https://github.com/horovod/horovod) is also supported for large-scale scenarios.
- **Reliable and reproducible benchmarks**: NeurST reports strong baselines with well-designed hyper-parameters on several benchmark datasets (MT&ST). It provides a series of recipes to reproduce them. 


## Pretrained Models & Performance Benchmarks
NeurST provides reference implementations of various models and benchmarks. Please see [examples](/examples) for model links and NeurST benchmark on different datasets.

- Text Translation
    - [Transformer on WMT14 en->de](/examples/translation)
- Speech-to-Text Translation
    - [libri-trans](/examples/speech_transformer/augmented_librispeech)
    - [MuST-C](/examples/speech_transformer/must-c)


## Requirements and Installation

- Python version >= 3.6
- TensorFlow >= 2.3.0

Install NeurST from source:
```
git clone https://github.com/bytedance/neurst.git
cd neurst/
pip3 install -e .
```
If there exists ImportError during running, manually install the required packages at that time.

## Citation
```
@InProceedings{zhao2021neurst,
  author       = {Chengqi Zhao and Mingxuan Wang and Qianqian Dong and Rong Ye and Lei Li},
  booktitle    = {the 59th Annual Meeting of the Association for Computational Linguistics (ACL): System Demonstrations},
  title        = {{NeurST}: Neural Speech Translation Toolkit},
  year         = {2021},
  month        = aug,
}
```

## Contact
Any questions or suggestions, please feel free to contact us: [zhaochengqi.d@bytedance.com](mailto:zhaochengqi.d@bytedance.com), [wangmingxuan.89@bytedance.com](mailto:wangmingxuan.89@bytedance.com).

## Acknowledgement
We thank Bairen Yi, Zherui Liu, Yulu Jia, Yibo Zhu, Jiaze Chen, Jiangtao Feng, Zewei Sun for their kind help. 

