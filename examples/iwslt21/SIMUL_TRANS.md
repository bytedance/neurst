# Simultaneous Translation systems for IWSLT 2021 

Here, we release our systems submitted to IWSLT2021 (**text-to-text**) and show how to evaluate the systems. For more details about the model structure and training datasets, see our [system report](https://arxiv.org/abs/2105.07319). Feel free to contact [wangtao.960826@bytedance.com](wangtao.960826@bytedance.com) if there is any question.
```
@inproceedings{zhao2021iwslt,
  author       = {Chengqi Zhao and Zhicheng Liu and Jian Tong and Tao Wang 
                    and Mingxuan Wang and Rong Ye and Qianqian Dong and Jun Cao and Lei Li},
  booktitle    = {Proceedings of the 18th International Conference on Spoken Language Translation},
  title        = {The Volctrans Neural Speech Translation System for IWSLT 2021},
  year         = {2021},
}

```

## Requirements
```bash
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval/
pip install -e .
```

## Results & Models
We report results of our models on [MuST-C V2](https://ict.fbk.eu/must-c/) tst-COMMON. 

### English-German (en2de)

```bash
# Download test data
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/tst-COMMON.en
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/tst-COMMON.de

# Download and untar model
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/iwslt21_simul_en2de_models.tgz
tar -zxvf iwslt21_simul_en2de_models.tgz

# Low Latency
python3 -m neurst.cli.simuleval_cli --agent simul_trans_text_agent \
    --model-dir ./all_1w-k7-big-last,./all_conv_1w-k7-big-avg,./part_1w-k7-big-avg/ \
    --data-type text --source tst-COMMON.en --target tst-COMMON.de \
    --num-processes 5 --output low \
    --wait-k 4 --force-segment 

# Medium Latency
python3 -m neurst.cli.simuleval_cli --agent simul_trans_text_agent \
    --model-dir ./all_1w-k7-big-last,./all_conv_1w-k7-big-avg,./part_1w-k7-big-avg/ \
    --data-type text --source tst-COMMON.en --target tst-COMMON.de \
    --num-processes 5 --output medium \
    --wait-k 10 --force-segment 

# High Latency
python3 -m neurst.cli.simuleval_cli --agent simul_trans_text_agent \
    --model-dir ./all_1w-k7-big-last,./all_conv_1w-k7-big-avg,./part_1w-k7-big-avg/ \
    --data-type text --source tst-COMMON.en --target tst-COMMON.de \
    --num-processes 5 --output high \
    --wait-k 13
```

|  Latency                 | BLEU  | AL    | AP   | DAL   |
|:------------------------:|:-----:|:-----:|:----:|:-----:|
| Low (k=4, force-seg) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/instances.log.low) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/scores.low)  | 28.78 | 2.86  | 0.69 | 4.22  |
| Medium (k=10, force-seg) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/instances.log.medium) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/scores.medium) | 32.88 | 5.80  | 0.83 | 9.05  |
| High (k=13) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/instances.log.high) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2de/scores.high) | 33.21 | 11.03 | 0.93 | 11.40 |


### English-Japanese (en2ja)

```bash
# Download test data
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/dev2021.en
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/dev2021.en

# Download and untar model
wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/iwslt21_simul_en2ja_models.tgz
tar -zxvf iwslt21_simul_en2ja_models.tgz

# Low Latency
python3 -m neurst.cli.simuleval_cli --agent simul_trans_text_agent \
    --model-dir all_8k-k300-big-last/,score4_8k-multipath-big-avg/ \
    --data-type text --source dev2021.en --target dev2021.ja \
    --sacrebleu-tokenizer ja-mecab --no-space --eval-latency-unit char \
    --num-processes 5 --output low \
    --wait-k 13 --force-segment 

# Medium Latency
#   same as low latency 

# High Latency
python3 -m neurst.cli.simuleval_cli --agent simul_trans_text_agent \
    --model-dir all_8k-k300-big-last/,score4_8k-multipath-big-avg/ \
    --data-type text --source dev2021.en --target dev2021.ja \
    --sacrebleu-tokenizer ja-mecab --no-space --eval-latency-unit char \
    --num-processes 5 --output high \
    --wait-k 13
```

|  Latency                 | BLEU  | AL    | AP   | DAL   |
|:------------------------:|:-----:|:-----:|:----:|:-----:|
| Low (k=13, force-seg) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/instances.log.low) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/scores.low)  | 15.80 | 6.34  | 0.89 | 11.14  |
| Medium (k=13, force-seg) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/instances.log.low) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/scores.low) | 15.80 | 6.34  | 0.89 | 11.14  |
| High (k=13) [[instance]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/instances.log.high) [[scores]](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/iwslt21/simul/en2ja/scores.high) | 15.84 | 11.19 | 0.97 | 11.81 |


## Official Blind Test (2021)

### English-German (en2de)

|  Latency                 | BLEU  | AL    | AP   | DAL   |
|:------------------------:|:-----:|:-----:|:----:|:-----:|
| Low (k=4, force-seg) | 23.24 | 3.08  | 0.68 | 4.25  |
| Medium (k=10, force-seg) | 27.22 | 6.30  | 0.81 | 9.24  |
| High (k=13) | 26.82 | 12.03 | 0.92 | 12.39 |


### English-Japanese (en2ja)
 
|  Latency                 | BLEU  | AL    | AP   | DAL   |
|:------------------------:|:-----:|:-----:|:----:|:-----:|
| Low (k=13, force-seg) | 16.91 | 6.54  | 0.89 | 11.26  |
| Medium (k=13, force-seg) | 16.91 | 6.54  | 0.89 | 11.26  |
| High (k=13) | 16.97 | 11.27 | 0.97 | 11.90 |
