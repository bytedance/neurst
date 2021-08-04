# Speech Translation on Argumented LibriSpeech

[Augmented LibriSpeech](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus) is a small EN->FR speech translation corpus which was originally started from the LibriSpeech corpus. The English utterances were automatically aligned to the e-books in French and 236 hours of English speech aligned to French translations at utterance level were finally extracted. It has been widely used in previous studies. As such, we use the clean 100-hour portion plus the augmented machine translation from Google Translate as the training data.

The final performance of speech translation on Argumented LibriSpeech is: 

> See [RESULTS](/examples/speech_transformer/augmented_librispeech/RESULTS.md) for the comparison with counterparts. 

- **ASR (dmodel=256, WER)** 

|Model|Dev|Test|
|---|---|---|
|Transformer ASR |8.8|8.8|
 

- **MT and ST (dmodel=256, case-sensitive, tokenized BLEU/detokenized BLEU)**

|Model|Dev|Test|
|---|---|---|
|Transformer MT |20.8 / 19.3 | 19.3 / 17.6 |
|cascade ST (Transformer ASR -> Transformer MT) | 18.3 / 17.0| 17.4 / 16.0 |
|Transformer ST + ASR pretrain | 18.3 / 16.9 | 16.9 / 15.5  |
|Transformer ST + ASR pretrain + SpecAug | 19.3 / 17.8 | 17.8 / 16.3  |
|Transformer ST ensemble above 2 models | **19.3** / **18.0** | **18.3 / 16.8**  |

- **MT and ST (dmodel=256, case-insensitive, tokenized BLEU/detokenized BLEU)**

|Model|Dev|Test|
|---|---|---|
|Transformer MT | 21.7 / 20.2 | 20.2 / 18.5 |
|cascade ST (Transformer ASR -> Transformer MT) | 19.2 / 17.8 | 18.2 / 16.8 |
|Transformer ST + ASR pretrain | 19.2 / 17.8 | 17.9 / 16.5 |
|Transformer ST + ASR pretrain + SpecAug | 20.2 / 18.7 | 18.7 / 17.2  |
|Transformer ST ensemble above 2 models | **20.3** / **18.9** | **19.2** / **17.7**  |

In this recipe, we will introduce how to pre-process the Augmented LibriSpeech corpus and train/evaluate a speech translation model using neurst.

### Contents
* [Requirements](#requirements)
* [Data preprocessing](#data-preprocessing)
    * [Step 1: Download Data](#step-1:-download-data)
    * [Step 2: Extract audio features](#step-2:-extract-audio-features)
    * [Step 3: Preprocess transcriptions and translations](#step-3-preprocess-transcriptions-and-translations)
* [Training and evaluation](#training-and-evaluation)
    * [Training with Validation](#training-with-validation)
    * [Accelerating Training with TensorFlow XLA](#accelerating-training-with-tensorflow-xla)
    * [Evaluation on Testset](#evaluation-on-testset)
    * [Training ST with ASR pretraining](#training-st-with-asr-pretraining)
    * [SpecAugment](#specaugment)
    * [Cascade ST](#cascade-st)
 
 
## Requirements

**apt**
- libsndfile1

**pip**
- TensorFlow >=2.3.0
- soundfile
- python_speech_features
- subword-nmt
- pyyaml
- sacrebleu
- sacremoses

**others**
```bash
$ git clone https://github.com/moses-smt/mosesdecoder.git
```


## Data preprocessing

### Step 1: Download Data
First, we [download](https://github.com/alicank/Translation-Augmented-LibriSpeech-Corpus) the original zip files into directory `/path_to_data/raw/` and we have
```bash
/path_to_data/
└── raw
    ├── train_100h.zip
    ├── dev.zip
    └── test.zip
```

### Step 2: Extract audio features
The speech translation corpus contains source raw audio files, texts in a target language and other optional information (e.g. transcriptions of the corresponding audio files). Here we pre-compute audio features (that is, log-mel filterbank coefficients) because the computation is time-consuming and features are usually fixed during training and evaluation.

Though NeurST supports preprocessing audio inputs on-the-fly, we recommend to pack the extracted features into TF Records to alleviate the I/O and CPU overhead.
 
We can extract audio features with 
```bash
$ ./examples/speech_to_text/augmented_librispeech/02-audio_feature_extraction.sh /path_to_data
``` 
By default, it extracts 80-channel log-mel filterbank coefficients using a lightweight python package [`python_speech_features`](https://github.com/jameslyons/python_speech_features) with windows of 25ms and steps of 10ms. Then we have
```bash
/path_to_data/
├── devtest
│   ├── dev.tfrecords-00000-of-00001
│   └── test.tfrecords-00000-of-00001
├── train
│   ├── train.tfrecords-00000-of-00064
│   ├── ......
│   └── train.tfrecords-00063-of-00064
└── transcripts
    ├── dev.en.txt
    ├── dev.fr.txt
    ├── test.en.txt
    ├── test.fr.txt
    ├── train.en.txt
    └── train.fr.txt
```
where the directory `/path_to_data/train/`(`/path_to_data/devtest`) contains the extracted audio features and the corresponding transcriptions (and translations) in TF Record format for training (and evaluation). Transcriptions and translations in txt format are stored in `/path_to_data/transcripts`.

Furthermore, to examine the elements in the TF Record files, we can simply run the command line tool `view_tfrecord`:
```bash
$ python3 -m neurst.cli.view_tfrecord /path_to_data/train/

features {
  feature {
    key: "audio"
    value {
      float_list {
        value: -0.3024393916130066
        value: -0.4108518660068512
        ......
      }
    }
  }
  feature {
    key: "transcript"
    value {
      bytes_list {
        value: "valentine"
      }
    }
  }
  feature {
    key: "translation"
    value {
      bytes_list {
        value: "Valentin?"
      }
    }
  }
}

elements: {
    "transcript": bytes (str)
    "translation": bytes (str)
    "audio": float32
}
```

### Step 3 Preprocess transcriptions and translations
As is mentioned above, we can map the word tokens to IDs aforehand, to speed up the training process.

By running with
```bash
$ ./examples/speech_to_text/augmented_librispeech/03-preprocess.sh /path_to_moses /path_to_data
```
we learn vocabulary based on [BPE](https://github.com/rsennrich/subword-nmt) rules with 8,000 merge operations. The learnt BPE and vocabulary are shared across ASR, MT and ST tasks. Note that, we lowercase the transcriptions and remove all punctuations while the cases and punctuations of translations are reserved and we simply apply [moses](https://github.com/moses-smt/mosesdecoder) tokenizer. As a result, we obtain 
```bash
/path_to_data/
├── asr_st
│   ├── asr_prediction_args.yml
│   ├── asr_training_args.yml
│   ├── asr_validation_args.yml
│   ├── codes.bpe
│   ├── st_prediction_args.yml
│   ├── st_training_args.yml
│   ├── st_validation_args.yml
│   ├── train
│   │   ├── train.tfrecords-00000-of-00064
│   │   ├── ......
│   │   └── train.tfrecords-00063-of-00064
│   ├── vocab.en
│   └── vocab.fr
└── mt
    ├── codes.bpe
    ├── mt_prediction_args.yml
    ├── mt_training_args.yml
    ├── mt_validation_args.yml
    ├── train
    │   ├── train.en.bpe.txt
    │   └── train.fr.tok.bpe.txt
    ├── vocab.en
    └── vocab.fr
```
Here, we use txt files (not TF Record) for MT tasks, while the pre-processed training samples for ASR/ST are stored in TF Record files (`/path_to_data/asr_st/train/`). 

In addition, configuration files (`*.yml`) are generated for the following training/evaluation process. In detail,
- `*_training_args.yml`: defines the arguments for training, such as batch size, optimizer, paths of training data and data pre-processing pipelines.
- `*_validation_args.yml`: defines the arguments for validation during training, containing validation dataset, interval between two validation procedures, metrics and configurations about automatic checkpoint average.
- `*_prediction_args.yml`: defines the arguments for inference and evaluation, containing testsets, inferece options (like beam size) and metric.

## Training and evaluation

### Training with validation
Let's take ASR as an example:
```bash
python3 -m neurst.cli.run_exp \
    --config_paths /path_to_data/asr_st/asr_training_args.yml,/path_to_data/asr_st/asr_validation_args.yml \
    --hparams_set speech_transformer_s \
    --model_dir /path_to_data/asr_st/asr_benchmark
```
where `/path_to_data/asr_st/asr_benchmark` is the root path for checkpoints. Here we use `--hparams_set speech_transformer_s` to train a transformer model including 12 encoder layers and 6 decoder layers with `dmodel=256`.
 > Alternatively, we can set `--hparams_set speech_transformer_m` to use the `dmodel=512` version, which usually achives better performance.

We train the ASR model on multiple GPUs, as long as there is no GPU out-of-memory exception. Moreover, we can set `--update_cycle n --batch_size 120000//n` to simulate `n` GPUs with 1 GPU.

### Accelerating training with TensorFlow XLA

To accelerate the training speed, we can simply enable TensorFlow XLA via `--enable_xla` option and separate the validation procedure from the training, that is
```bash
python3 -m neurst.cli.run_exp \
    --config_paths /path_to_data/asr_st/asr_training_args.yml \
    --hparams_set speech_transformer_s \
    --model_dir /path_to_data/asr_st/asr_benchmark \
    --enable_xla
```

Then, we start another process with one GPU for validation by
```bash
python3 -m neurst.cli.run_exp \
    --entry validation \
    --config_paths /path_to_data/asr_st/asr_training_args.yml \
    --model_dir /path_to_data/asr_st/asr_benchmark
```

This process will constantly scan the `model_dir`, evaluate each checkpoint and store the checkpoints with best metrics (e.g. WER for ASR) into `{model_dir}/best` directory along with the corresponding averaged version (by default 10 latest checkpoints) into `{model_dir}/best_avg`. 

### Evaluation on Testset
By running with
```bash
python3 -m bytdseq.cli.run_exp \
    --config_paths /path_to_data/asr_st/asr_prediction_args.yml \
    --model_dir /path_to_data/asr_st/asr_benchmark/best_avg
```
WER will be reported on both dev and test set.

One can replace the yaml files and model directory of ASR with MT/ST's to train and evaluate MT/ST models. 


### Training ST with ASR Pretraining

In ST literature, training ST is more difficult than ASR and MT. Transfer learning from ASR and MT tasks is an effective approach to this problem. To do so, we can initialize the ST encoder with ASR encoder by two additional options for training:
```bash
    --pretrain_model /path_to_data/asr_st/asr_benchmark/best_avg \
    --pretrain_variable_pattern "(TransformerEncoder)|(input_audio)"
```
The variables that match the regular expression provided by `--pretrain_variable_pattern` will be initialized. 

On this basis, we can further initialize the ST decoder with MT decoder by following options:
```bash
    --pretrain_model /path_to_data/asr_st/asr_benchmark/best_avg /path_to_data/mt/mt_benchmark/best_avg \
    --pretrain_variable_pattern "(TransformerEncoder)|(input_audio)" "(TransformerDecoder)|(target_symbol)"
``` 
> To inspect the names of model variables, use `inspect_checkpoint` tool (see [neurst/cli/README.md](/neurst/cli/README.md)).  


### SpecAugment
To further improve the performance of ASR or ST, we can apply SpecAugment (Park et al., 2019) by option `--specaug VALUE`. Alternatively, the VALUE can be set to LB, LD, SM and SS (described in the original paper), or a json-like string defining the detailed arguments (see [neurst/utils/audio_lib.py](/neurst/utils/audio_lib.py)))


### Cascade ST
NeurST provides `cascade_st` tool for easily combining ASR and MT models, e.g.

```bash
python3 -m neurst.cli.cascade_st \
    --dataset AudioTripleTFRecordDataset
    --dataset.params "{'data_path':'/path_to_data/devtest/test.tfrecords-00000-of-00001'}" \
    --asr_model_dir /path_to_data/asr_st/asr_benchmark/best_avg \
    --asr_search_method beam_search \
    --asr_search_method.params "{'beam_size':4,'length_penalty':-1,'maximum_decode_length':150}" \
    --mt_model_dir /path_to_data/mt/mt_benchmark/best_avg \
    --mt_search_method beam_search \
    --mt_search_method.params "{'beam_size':4,'length_penalty':-1,'maximum_decode_length':180}" 
```

For more details about the arguments, use `-h` option.
