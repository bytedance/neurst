# GigaST: A large-scale speech translation corpus

GigaST is a large-scale speech translation corpus, by translating the transcriptions in GigaSpeech, a multi-domain English speech recognition corpus with 10,000 hours of labeled audio. The training data is translated by a strong machine translation system and the test data is produced by professional human translators.

### Contents
* [Download](#download)
* [Dataset](#dataset)
    * [Audio Source](#audio-source)
    * [Training Set](#training-set)
    * [Test Sets](#test-sets)
* [Preparation Guidelines](#preparation-guidelines)
* [Benchmarks](#benchmarks)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)

## Download 

The GigaST dataset can be downloaded from: 


| Language | Version | Link |
|:---:|:---:|:---:|
|En-De|v1.0.0| [GigaST.de.json](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/datasets/GigaST/GigaST.de.json)|
|En-Zh|v1.0.0| [GigaST.zh.json](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/datasets/GigaST/GigaST.zh.json)|



The corresponding audio recordings and transcriptions can be found in [*GigaSpeech*](https://github.com/SpeechColab/GigaSpeech). 

## Dataset

### Audio Source

The audio recordings are not included in the released files but can be found in [*GigaSpeech*](https://github.com/SpeechColab/GigaSpeech). So we also follow the partitioning of the dataset, that is,
> | Subseet | Hours | Remarks |
> |:---:|:---:|:---|
> | XS | 10 | System building and debugging |
> | S | 250 | Quick research experiments |
> | M | 1,000 | Large-scale research experiments |
> | L | 2,500 | Medium-scale industrial experiments |
> | XL | 10,000 | Large-scale industrial experiments | 

### Training Set

The labels (translations) of GigaST are produced by strong MT systems. To verify the quality, we conduct human evalutions on the pseudo labels and we present results in [paper](https://arxiv.org/abs/2204.03939).

The detailed statistics of training set is listed below. Here we remove all non-speech segments, such as music and noise. The #tokens is counted in character-level for En-Zh.

**En-Zh**

| Subset | #seg. | #hours | #tokens |
|:---:|:---:|:---:|:---:|
| S | 210,012 | 243.1 | 4.1M |
| M | 835,846 | 974.3 | 16.7M |
| L | 2,084,274 | 2,337.7 | 41.8M |
| XL | 7,650,889 | 9,780.8 | 168.3M | 

**En-De**

| Subset | #seg. | #hours | #tokens |
|:---:|:---:|:---:|:---:|
| S | 221,572 | 256.2 | 2.5M |
| M | 868,316 | 1,013.1 | 10.2M |
| L | 2,147,471 | 2,510.9 | 25.3M |
| XL | 7,815,436 | 9,997.9 | 101.6M |

### Test sets

The test sets are produced by professional human translators based on *GigaSpeech*'s test set. Note that the En-De test set contains a subset of it for this current release and, a small number of transcriptions are difficult to understand due to the lack of context and are ignored.

The detailed statistics of test sets is listed below.

| Language | #seg. | #hours | #tokens |
|:---:|:---:|:---:|:---:|
| En-Zh | 19,888 | 35.3 | 637.8K |
| En-De | 4,163 | 7.1 | 69.2K |


## Preparation Guidelines

The released GigaST data files only contain `sid`, `text_raw` and the alignment score (range from 0 to 10) with the original transcription. A snap of the data file is:
```json
{
  "dataset": "GigaST",
  "description": "......",
  "language": "DE",
  "version": "v1.0.0",
  "audios": [
    {
      "segments": [
        {
          "sid": "POD0000000001_S0000008",
          "text_raw": "Douglas McGray wird unser F\u00fchrer sein, du gehst durch die T\u00fcr, du siehst den roten Teppich, du siehst jemanden im Anzug. sie gr\u00fc\u00dfen dich vielleicht.",
          "extra": {
            "alignment_score": 8.21
          }
        },
        ... ...
      ],
      ... ...
    },
    ... ...
  ]
}

```

We provide [convert_data.py](/examples/speech_transformer/gigast/convert_data.py) script to convert the data files to *GigaSpeech*'s data format. Then, one can conveniently reuse those preparation scripts designed for *GigaSpeech*. 

```bash
python3 examples/speech_transformer/gigast/convert_data.py \
    --gigaspeech_file GigaSpeech.json \
    --gigast_file GigaST.zh/de.json \
    --output_file output.json
```

A snap of the output is:
```json
{
  "dataset": "GigaST",
  "description": "......",
  "language": "DE",
  "version": "v1.0.0",
  "audios": [
    {
      "title": "Check Cashing Stores",
      "path": "audio/podcast/P0001/POD0000000001.opus",
      "aid": "POD0000000001",
      ... ...
      "segments": [
        {
          "sid": "POD0000000001_S0000008",
          "begin_time": 159.0,
          "end_time": 167.52,
          "text_tn": "Douglas McGray wird unser F\u00fchrer sein, du gehst durch die T\u00fcr, du siehst den roten Teppich, du siehst jemanden im Anzug. sie gr\u00fc\u00dfen dich vielleicht.",
          "subsets": [
              "{XL}"
          ],
          "extra": {
            "alignment_score": 8.21
          }
        },
        ... ...
      ],
      ... ...
    },
    ... ...
  ]
}
```

## Benchmarks

We provide training recipes of end-to-end speech translation models using GigaST dataset. The scripts, models and benchmarks can be found in [examples/speech_transformer/gigast](/examples/speech_transformer/gigast).


## Citation

```
@Article{gigast,
  author  = {Ye, Rong and Zhao, Chengqi and Ko, Tom and Meng, Chutong and Wang, Tao and Wang, Mingxuan and Cao, Jun},
  journal = {arXiv preprint arXiv:2204.03939},
  title = {GigaST: A 10,000-hour Pseudo Speech Translation Corpus},
  year    = {2022},
}
```


## Acknowledgement
*GigaSpeech* dataset is essential for the creation of GigaST. We are extremely grateful to the *GigaSpeech*'s contributors.

