# Official NeurST Examples and Modules

The folder contains example (re-)implementations of selected research papers and benchmarks, with released model checkpoints.

## Speech-to-Text Translation

### 2021

- [ACL demo] Zhao et al. NeurST: Neural Speech Translation Toolkit. [[paper](https://aclanthology.org/2021.acl-demo.7/)]
    - E2E ST benchmark: [libri-trans](/examples/speech_transformer/augmented_librispeech), [must-c](/examples/speech_transformer/must-c)

- [IWLST 2021 System] Zhao et al. The Volctrans Neural Speech Translation System for IWSLT 2021. [[paper](https://aclanthology.org/2021.iwslt-1.6/)]
    - [Offline ST](/examples/iwslt21/OFFLINE.md) 
    - [Simultaneous Translation](/examples/iwslt21/SIMUL_TRANS.md)


## Neural Machine Translation

### 2021

- [AAAI] Liang et al. Finding Sparse Structures for Domain Specific Neural Machine Translation. [[paper](https://arxiv.org/abs/2012.10586)][[example](/examples/prune_tune)]
 

### 2020

- [AAAI] Yang et al. Towards Making the Most of BERT in Neural Machine Translation. [[paper](https://arxiv.org/abs/1908.05672)][[example](/examples/ctnmt)]

### 2019

- [ICLR] Wu et al. Pay Less Attention With Lightweight and Dynamic Convolutions. [[paper](https://arxiv.org/pdf/1901.10430.pdf)][code only]


### 2017

- [NIPS] Vaswani et al. Attention Is All You Need. [[paper](https://arxiv.org/pdf/1706.03762.pdf)]
    - MT benchmark: [WMT14 EN->DE](/examples/translation)

## Neural Network Techniques

- Weight Pruning
    - unstructured pruning [[example](/examples/weight_pruning)]
- Quantization
    - quantization aware training [[ref](https://arxiv.org/abs/1712.05877)][[example](/examples/quantization)]
