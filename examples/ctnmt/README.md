# CTNMT (Yang et al., 2020)

In this code repo, we reproduced the three techniques proposed in the CTNMT paper to use BERT to train the NNT model. Feel free to contact [zhuyaoming@bytedance.com](zhuyaoming@bytedance.com) if there is any question.  

### Citation 
```bibtex
@inproceedings{yang2020towards,
  title={Towards making the most of bert in neural machine translation},
  author={Yang, Jiacheng and Wang, Mingxuan and Zhou, Hao and Zhao, Chengqi and Zhang, Weinan and Yu, Yong and Li, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={9378--9385},
  year={2020}
}
```

## Training the model 

1. Download our pre-processed BERT models and the corresponding vocabulary lists (will be released soon).
- here we assume your BERT models are saved at /tmp/BERT_BASE/ and /tmp/BERT_LARGE/, your data is saved at /tmp/data/

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


