# NeurST: Neural Speech Translation Toolkit
NeurST aims at easily building and training end-to-end speech translation, which has the careful design for extensibility and scalability. We believe this design can make it easier for NLP researchers to get started. In addition, NeurST allows researchers to train custom models for translation, summarization and so on.

> NeurST is based on TensorFlow2 and we are working on the pytorch version.

## Features

### Models
NeurST provides reference implementations of various models, including:

- Transformer (self-attention) networks
    - [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/pdf/1706.03762.pdf)
    - [Pay Less Attention With Lightweight and Dynamic Convolutions (Wu et al., 2019)](https://arxiv.org/pdf/1901.10430.pdf)

- comming soon...

### Recipes and Benchmarks
NeurST provides several **strong and reproducible benchmarks** for various tasks:

- Translation
    - [Transformer models on WMT14 en->de](/examples/translation)
- Speech-to-Text
    - [Augmented Librispeech](/examples/speech_to_text/augmented_librispeech)
    - [MuST-C](/examples/speech_to_text/must-c)
- [Weight Pruning](/examples/weight_pruning/README.md)
- [Quantization Aware Training](/examples/quantization/README.md) 

### Additionally

- multi-GPU (distributed) training on one machine or across multiple machines
    - `MirroredStrategy` / `MultiWorkerMirroredStrategy`
    - [`Byteps`](https://github.com/bytedance/byteps) / [`Horovod`](https://github.com/horovod/horovod)
- mixed precision training (trains faster with less GPU memory)
- multiple search algorithms implemented:
    - beam search
    - sampling (unconstrained, top-k and top-p)
- large mini-batch training even on a single GPU via delayed updates (gradient accumulation)
- TensorFlow savedmodel for TensorFlow-serving
- TensorFlow XLA support for speeding up training
- extensible: easily register new datasets, models, criterions, tasks, optimizers and learning rate schedulers

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
@misc{zhao2020neurst,
      title={NeurST: Neural Speech Translation Toolkit}, 
      author={Chengqi Zhao and Mingxuan Wang and Lei Li},
      year={2020},
      eprint={2012.10018},
      archivePrefix={arXiv},
}
```

## Contact
Any questions or suggestions, please feel free to contact us: [zhaochengqi.d@bytedance.com](mailto:zhaochengqi.d@bytedance.com), [wangmingxuan.89@bytedance.com](mailto:wangmingxuan.89@bytedance.com).

## Acknowledgement
We thank Bairen Yi, Zherui Liu, Yulu Jia, Yibo Zhu, Jiaze Chen, Jiangtao Feng, Zewei Sun for their kind help. 
