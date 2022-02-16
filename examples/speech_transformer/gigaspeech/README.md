# ASR on GigaSpeech

[GigaSpeech](https://github.com/SpeechColab/GigaSpeech) is a large English speech recognition corpus collected from audiobooks, podcasts, and Youtube. GigaSpeech contains 5 subsets of different sizes for difference usages. We use the largest subset XL which has 10,000 hours of audio as the training data for ASR.

The final performance of ASR on GigaSpeech is

> See [RESULTS](/examples/speech_transformer/gigaspeech/RESULTS.md) for the comparison with counterparts.

The benchmark models:
| Task | Language | Models | Hypothesis | Reference |
|---|---|---|---|---|
|ASR|EN|[speech_transformer_l](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ckpt.tgz)| [Dev](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/hypo_dev.txt) [Test](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/hypo_test.txt)| [Dev](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ref_dev.txt) [Test](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ref_test.txt)|

- **ASR (dmodel=1024, WER)**

|Model|Dev|Test|
|---|---|---|
|speech_transformer_l |11.89|11.60|

In this recipe, we will introduce how to pre-process the GigaSpeech corpus and train/evaluate an ASR model using neurst.

### Contents
* [Requirements](#requirements)
* [Data preprocessing](#data-preprocessing)
    * [Step 1: Download Data](#step-1-download-data)
    * [Step 2: Extract audio features and Preprocess transcripts](#step-2-extract-audio-features-and-preprocess-transcripts)
* [Training and evaluation](#training-and-evaluation)

## Requirements

**apt**
- libsndfile1
- ffmpeg

**pip**
- TensorFlow >=2.4.1
- soundfile
- python_speech_features
- pyyaml
- sentencepiece


## Data preprocessing
### Step 1 Download Data
First, we [download](https://github.com/SpeechColab/GigaSpeech) the original opus files into directory `/path_to_data/` and we have
```bash
/data_to_path/
  ├── audio
  │   ├── audiobook
  │   │   ├── P0001
  │   │   │   ├── AUD0000000001.opus
  │   │   │   ├── AUD0000000002.opus
  │   │   │   ├── ..................
  │   │   ├── ...
  │   ├── podcast
  │   │   ├── P0000
  │   │   ├── ...
  │   └── youtube
  │       ├── P0000
  │       ├── ...
  └── GigaSpeech.json
```
These files take about **450GB**. Please make sure your disk has enough space.

### Step 2: Extract audio features and Preprocess transcripts
The ASR corpus contains source raw audio files and text files in English. Here we compute audio features (log-mel filterbank coefficients) and map the transcript tokens to IDs.

This step is extremely time comsuing because of the size of the original corpus. Also, the resultant TFRecords take about **1.1TB**. Please make sure your disk has enough space.

We can extract audio features and transcripts with
```bash
$ ./examples/speech_transformer/gigaspeech/./02-preprocess.sh /path_to_data subset 
``` 
If you want to train ASR with punctuations reserved, you can add `--keep-punctuation` option. By default, we ignore all punctuations during training, validation, and testing.

By default, it extracts 80-channel log-mel filterbank coefficients using a lightweight python package [`python_speech_features`](https://github.com/jameslyons/python_speech_features) with windows of 25ms and steps of 10ms. Then we have
```bash
/path_to_data/
├── spm.model
├── spm.vocab
└── asr
    ├── asr_data_prep.yml
    ├── asr_prediction_args.yml
    ├── asr_training_args.yml
    ├── asr_validation_args.yml
    ├── train
    │   ├── train.tfrecords-00000-of-01664
    │   ├── ......
    │   └── train.tfrecords-01663-of-01664
    └── devtest
        ├── DEV.tfrecords-00000-of-00001
        └── TEST.tfrecords-00000-of-00001
```
where the directory `/path_to_data/asr/train` (`/path_to_data/asr/devtest`) contains the extracted audio features and the corresponding transcriptions in TFRecord format for training (and evaluation). 

Furthermore, to examine the elements in the TFRecord files, we can simply run the command line tool `view_tfrecord`:
```bash
$ python3 -m neurst.cli.view_tfrecord /path_to_data/asr/train/

features {
  feature {
    key: "audio"
    value {
      float_list {
        ......
        value: -0.12725864350795746
        value: -0.30450382828712463
        value: -0.3461953103542328
        value: -0.3424985408782959
        value: -0.5286926031112671
      }
    }
  }
  feature {
    key: "audio_length"
    value {
      int64_list {
        value: 4
      }
    }
  }
  feature {
    key: "transcript"
    value {
      int64_list {
        value: 5
        value: 72
        value: 36
        value: 8
        value: 4
        value: 6720
        value: 54
        value: 182
        value: 9
        value: 7560
        value: 45
        value: 476
        value: 10002
      }
    }
  }
}

elements: {
    "audio": float32
    "audio_length": int64
    "transcript": int64
}
```

## Training and evaluation

The training and evaluation procedures are the same as those of [AugmentedLibrispeech](/examples/speech_to_text/augmented_librispeech/README.md).

Specifically, if you are training the XL subset, please use the `speech_transformer_l` model. 

Also, please use the default arguments in `asr_training_args.yml` if possible, especially the `batch_size`. If you train a large model with a small `batch_size`, e.g. `speech_transformer_l` with batch_size=20,000, the training procedure is hard to converge.