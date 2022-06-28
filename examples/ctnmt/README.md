# Utilizing BERT in Neural Machine Translation

Can we utilize extremely large monolingual text to improve neural machine translation without the expensive back-translation? 
Neural machine translation models are trained on parallel bilingual corpus. Even the large ones only include 20 to 40 millions of parallel sentence pairs. 
In the meanwhile, pre-trained language models such as BERT and GPT are trained on usually billions of monolingual sentences. 
Direct use BERT as the initialization for Transformer encoder could not gain any benefit, due to the catastrophic forgetting problem of BERT knowledge during further training on MT data. 
This example shows how to run the [CTNMT](https://arxiv.org/abs/1908.05672) (Yang et al. 2020) training method that integrates BERT into a Transformer MT model, the first successful method to do so. 

The trained checkpoint is available [here](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/aaai2020/ctnmt/ckpt.ctnmt.zip) or at [Zenodo](https://zenodo.org/record/6766335#.YrqanuxBxTY) .


The CTNMT method is from the following paper: Towards making the most of BERT in neural machine translation.
```bibtex
@inproceedings{yang2020towards,
  title={Towards making the most of BERT in neural machine translation},
  author={Yang, Jiacheng and Wang, Mingxuan and Zhou, Hao and Zhao, Chengqi and Zhang, Weinan and Yu, Yong and Li, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={9378--9385},
  year={2020}
}
```

## Results on WMT benchmark

Dataset: WMT 14 English - German
Evaluation metric: token BLEU
(notice some paper use a different way to measure the BLEU)

| Model                                   | baseline | + Rate scheduling | Dynamic switch | Asymptotic Distillation | Full CTNMT |
|-----------------------------------------|----------|-------------------|----------------|-------------------------|-------------|
| Transformer(hidden=768, enc=12, dec=6)  | 28.3     | 30.0              | 29.9           | 28.7                    |  30.1        |
| Transformer(hidden=1024, enc=6, dec=6)  | 28.4     | 29.6              | 28.9           | 28.6                    |            |
| Transformer(hidden=1024, enc=12, dec=6) | 29.4     | 30.5              | 30.5           | 29.0                    |            |


## Training the model 

1. Download our pre-processed BERT models and the corresponding vocabulary lists.
+ [BERT_LARGE](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/ctnmt/BERT_LARGE.zip)
+ [BERT_BASE](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/ctnmt/BERT_BASE.zip)
+ [vocabulary lists](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/ctnmt/vocab.zip)

You can also download the [trained CTNMT](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/aaai2020/ctnmt/ckpt.ctnmt.zip)

     we assume your BERT models are saved at /tmp/BERT_BASE/ and /tmp/BERT_LARGE/, your data is saved at /tmp/data/

2. run the command:

```bash
python3 -m neurst.cli.run_exp \
  --config_paths "${THE_CONFIG_PATH}"
```

* The example configurations can be found in `examples/ctnmt/example_configs`

### Important arguments
```yaml
# enable asymptotic distillation (note bert mode shall be "dynamic_switch" or "bert_distillation")
# we find that setting alpha=0.99 makes a better overall results 
criterion.class: LabelSmoothedCrossEntropyWithKd
criterion.params:
  kd_weight: 0.01

# enable dynamic switch
model.params:
  bert_mode: "dynamic_switch"

# enable Rate Scheduled
# According to https://openreview.net/forum?id=SJx37TEtDH , Adam is more friendly to Transformer compared to SGD
optimizer.class: Adam
optimizer_controller: RateScheduledOptimizer
optimizer_controller_args:
  warm_steps: 10000
  freeze_steps: 20000
  controlled_varname_pattern: bert

# evaluation
validator.params:
  eval_metric.class: CompoundSplitBleu


```


Feel free to contact [zhuyaoming@bytedance.com](zhuyaoming@bytedance.com) if there is any question.  

