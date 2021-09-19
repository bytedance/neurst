# Speech Translation on MuST-C

[MuST-C](https://ict.fbk.eu/must-c/) is a multilingual speech translation corpus whose size and quality facilitates the training of end-to-end systems for speech translation from English into several languages. For each target language, MuST-C comprises several hundred hours of audio recordings from English TED Talks, which are automatically aligned at the sentence level with their manual transcriptions and translations.

The final performance of speech translation on 8 languages of MuST-C (tst-COMMON) is: 

> See [RESULTS](/examples/speech_transformer/must-c/RESULTS.md) for the comparison with counterparts. 

The benchmark models:

|Language| Models|
|---|---|
|DE | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/de/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/de/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/de/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/de/st_specaug.tgz)\] |
|ES | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/es/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/es/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/es/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/es/st_specaug.tgz)\] |
|FR | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/fr/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/fr/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/fr/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/fr/st_specaug.tgz)\] |
|IT | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/it/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/it/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/it/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/it/st_specaug.tgz)\] |
|NL | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/nl/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/nl/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/nl/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/nl/st_specaug.tgz)\] |
|PT | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/pt/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/pt/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/pt/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/pt/st_specaug.tgz)\] |
|RO | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ro/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ro/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ro/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ro/st_specaug.tgz)\] |
|RU | \[[ASR](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ru/asr.tgz)\] \[[MT](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ru/mt.tgz)\] \[[ST](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ru/st.tgz)\] \[[ST+SpecAug](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/mustc/ru/st_specaug.tgz)\] |


- **ASR (dmodel=256, WER)** 

|Model|DE|ES|FR|IT|NL|PT|RO|RU|
|---|---|---|---|---|---|---|---|---|
|Transformer ASR |13.6|13|12.9|13.5|13.8|14.4|13.7|13.4|
 

- **MT and ST (dmodel=256, case-sensitive, tokenized BLEU/detokenized BLEU)**

|Model|DE|ES|FR|IT|NL|PT|RO|RU|
|---|---|---|---|---|---|---|---|---|
|Transformer MT |27.9/27.8|32.9/32.8|42.2/40.2|29.0/28.5|32.9/32.7|34.4/34.0|27.5/26.4|19.3/19.1|
|cascade ST (Transformer ASR -> Transformer MT) |23.5/23.4|28.1/28.0|35.8/33.9|24.3/23.8|27.3/27.1|28.6/28.3|23.3/22.2|16.2/16.0|
|Transformer ST + ASR pretrain |21.9/21.9|26.9/26.8|34.2/32.3|22.6/22.2|26.5/26.4|27.8/27.6|21.9/20.9|15.0/15.2|
|Transformer ST + ASR pretrain + SpecAug |22.8/22.8|27.5/27.4|35.2/33.3|23.4/22.9|27.4/27.2|29.0/28.7|23.2/22.2|15.2/15.1|

In this recipe, we will introduce how to pre-process the MuST-C corpus and train/evaluate a speech translation model using neurst.

### Contents
* [Requirements](#requirements)
* [Data preprocessing](#data-preprocessing)
    * [Step 1: Download Data](#step-1:-download-data)
    * [Step 2: Extract audio features](#step-2:-extract-audio-features)
    * [Step 3: Preprocess transcriptions and translations](#step-3-preprocess-transcriptions-and-translations)
* [Training and evaluation](#training-and-evaluation)
    
 
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

Take the English-German portion as an example.

### Step 1: Download Data
First, we [download](https://ict.fbk.eu/must-c/) the original tgz files into directory `/path_to_data/raw/` and we have
```bash
/path_to_data/
└── raw
    ├── MUSTC_v1.0_en-de.tar.gz
    ├── MUSTC_v1.0_en-es.tar.gz
    └── ......
```

### Step 2: Extract audio features
The speech translation corpus contains source raw audio files, texts in a target language and other optional information (e.g. transcriptions of the corresponding audio files). Here we pre-compute audio features (that is, log-mel filterbank coefficients) because the computation is time-consuming and features are usually fixed during training and evaluation.

Though NeurST supports preprocessing audio inputs on-the-fly, we recommend to pack the extracted features into TF Records to alleviate the I/O and CPU overhead.
 
We can extract audio features with 
```bash
$ ./examples/speech_to_text/must-c/./02-audio_feature_extraction.sh /path_to_data de --untar
``` 
> Here, we use `--untar` option to first extract the tgz file, because it is quite time-consuming when we repeatedly iterate on the compressed file with a huge .h5 file inside.

By default, it extracts 80-channel log-mel filterbank coefficients using a lightweight python package [`python_speech_features`](https://github.com/jameslyons/python_speech_features) with windows of 25ms and steps of 10ms. Then we have
```bash
/path_to_data/
├── devtest
│   ├── dev.en-de.tfrecords-00000-of-00001
│   └── tst-COMMON.en-de.tfrecords-00000-of-00001
├── train
│   └── de
│       ├── train.tfrecords-00000-of-00128
│       ├── ......
│       └── train.tfrecords-00127-of-00128
└── transcripts
    └── de
        ├── dev.de.txt
        ├── dev.en.txt
        ├── train.de.txt
        ├── train.en.txt
        ├── tst-COMMON.de.txt
        └── tst-COMMON.en.txt
```
where the directory `/path_to_data/train/de/`(`/path_to_data/devtest`) contains the extracted audio features and the corresponding transcriptions (and translations) in TF Record format for training (and evaluation). Transcriptions and translations in txt format are stored in `/path_to_data/transcripts/de/`.

Furthermore, to examine the elements in the TF Record files, we can simply run the command line tool `view_tfrecord`:
```bash
$ python3 -m neurst.cli.view_tfrecord /path_to_data/train/de/

features {
  feature {
    key: "audio"
    value {
      float_list {
        ......
        value: -0.033860281109809875
        value: -0.025679411366581917
      }
    }
  }
  feature {
    key: "transcript"
    value {
      bytes_list {
        value: "She took our order, and then went to the couple in the booth next to us, and she lowered her voice so much, I had to really strain to hear what she was saying."
      }
    }
  }
  feature {
    key: "translation"
    value {
      bytes_list {
        value: "Sie nahm unsere Bestellung auf, ging dann zum Paar in der Nische neben uns und senkte ihre Stimme so sehr, dass ich mich richtig anstrengen musste, um sie zu verstehen."
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
MOSES_DIR ROOT_DATA_PATH TRG_LANG
```bash
$ ./examples/speech_to_text/must-c/03-preprocess.sh /path_to_moses /path_to_data de
```
we learn vocabulary based on [BPE](https://github.com/rsennrich/subword-nmt) rules with 8,000 merge operations. The learnt BPE and vocabulary are shared across ASR, MT and ST tasks. Note that, we lowercase the transcriptions and remove all punctuations while the cases and punctuations of translations are reserved and we simply apply [moses](https://github.com/moses-smt/mosesdecoder) tokenizer. As a result, we obtain 
```bash
/path_to_data/
├── asr_st
│   └── de
│       ├── asr_prediction_args.yml
│       ├── asr_training_args.yml
│       ├── asr_validation_args.yml
│       ├── codes.bpe
│       ├── st_prediction_args.yml
│       ├── st_training_args.yml
│       ├── st_validation_args.yml
│       ├── train
│       │   ├── train.tfrecords-00000-of-00064
│       │   ├── ......
│       │   └── train.tfrecords-00127-of-00128
│       ├── vocab.en
│       └── vocab.de
└── mt
    └── de
        ├── codes.bpe
        ├── mt_prediction_args.yml
        ├── mt_training_args.yml
        ├── mt_validation_args.yml
        ├── train
        │   ├── train.en.clean.tok.bpe.txt
        │   └── train.de.tok.bpe.txt
        ├── vocab.en
        └── vocab.de
```
Here, we use txt files (not TF Record) for MT tasks, while the pre-processed training samples for ASR/ST are stored in TF Record files (`/path_to_data/asr_st/de/train/`). 

In addition, configuration files (`*.yml`) are generated for the following training/evaluation process. In detail,
- `*_training_args.yml`: defines the arguments for training, such as batch size, optimizer, paths of training data and data pre-processing pipelines.
- `*_validation_args.yml`: defines the arguments for validation during training, containing validation dataset, interval between two validation procedures, metrics and configurations about automatic checkpoint average.
- `*_prediction_args.yml`: defines the arguments for inference and evaluation, containing testsets, inferece options (like beam size) and metric.

## Training and evaluation

The training and evaluation procedures are the same as those of [AugmentedLibrispeech](/examples/speech_transformer/augmented_librispeech/README.md).
