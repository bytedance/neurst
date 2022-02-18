# ASR on GigaSpeech

[GigaSpeech](https://github.com/SpeechColab/GigaSpeech) is a large English speech recognition corpus collected from audiobooks, podcasts, and Youtube. GigaSpeech contains 5 subsets of different sizes for difference usages. We use the largest subset XL which has 10,000 hours of audio as the training data for ASR.

In this recipe, we will introduce how to pre-process the GigaSpeech corpus and how to train/evaluate an ASR model using neurst. Also, we give an example of how to use our benchmark model for prediction.

### Contents
* [Requirements](#requirements)
* [Results and Models](#results-and-models)
    * [Results](#results)
    * [Models](#models)
* [Data preprocessing](#data-preprocessing)
    * [Step 1: Download Data](#step-1-download-data)
    * [Step 2: Extract audio features and preprocess transcripts for training set](#step-2-extract-audio-features-and-preprocess-transcripts-for-training-set)
    * [Step 3: Extract audio features and transcripts for dev and test set](#step-3-extract-audio-features-and-transcripts-for-dev-and-test-set)
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

## Results and Models
### Results
The final performance of ASR on GigaSpeech is

> See [RESULTS](/examples/speech_transformer/gigaspeech/RESULTS.md) for the comparison with counterparts.

- **ASR (dmodel=1024, WER)**

|Model|Dev|Test|
|---|---|---|
|speech_transformer_l |11.89|11.60|

### Models
The benchmark models:
| Task | Language | Models | Hypothesis | Reference |
|---|---|---|---|---|
|ASR|EN|[speech_transformer_l](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ckpt.tgz)| [Dev](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/hypo_dev.txt) [Test](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/hypo_test.txt)| [Dev](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ref_dev.txt) [Test](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ref_test.txt)|

An example of how to use our benchmark models for inference:
```bash
# create dev and test set
./03-create_devtest_set.sh /path_to_data/
# download and untar models
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/speech_to_text/gigaspeech/ckpt.tgz
tar -zxvf ckpt.tgz

# show WER on test and dev
python3 -m neurst.cli.run_exp \
  --config_paths /path_to_data/asr/asr_prediction_args.yml \
  --model_dir giga_xl

# save hypothesis to file
python3 -m neurst.cli.run_exp \
  --config_paths /path_to_data/asr/asr_prediction_args.yml \
  --model_dir giga_xl \
  --output_file "{'dev': 'dev.hypo.txt', 'test': 'test.hypo.txt'}"
```

## Data preprocessing
### Step 1 Download Data
First, we [download](https://github.com/SpeechColab/GigaSpeech) the original opus files into directory `/path_to_data/` and we have
```bash
/path_to_data/
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

### Step 2: Extract audio features and preprocess transcripts for training set
The ASR corpus contains source raw audio files and text files in English. Here we compute audio features (log-mel filterbank coefficients) and map the transcript tokens to IDs.

This step is extremely time comsuing because of the size of the original corpus. Also, the resultant TFRecords take about **1.1TB**. Please make sure your disk has enough space.

We can extract audio features and transcripts with
```bash
$ ./examples/speech_transformer/gigaspeech/./02-create_training_set.sh /path_to_data subset 
``` 
If you want to train ASR with punctuations reserved, you can add `--keep-punctuation` option. By default, we ignore all punctuations during training, validation, and testing.

By default, it extracts 80-channel log-mel filterbank coefficients using a lightweight python package [`python_speech_features`](https://github.com/jameslyons/python_speech_features) with windows of 25ms and steps of 10ms. Then we have
```bash
/path_to_data/
├── spm.model
├── spm.vocab
└── asr
    ├── asr_data_prep.yml
    ├── asr_training_args.yml
    └── train
        ├── train.tfrecords-00000-of-01664
        ├── ......
        └── train.tfrecords-01663-of-01664
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
### Step 3: Extract audio features and transcripts for dev and test set
This step is similar to step 2. The only difference is we do not convert transcript tokens to IDs. Instead, we use raw text for dev and test set in order to compute WER.

We can extract audio features and raw text with
```bash
$ ./examples/speech_transformer/gigaspeech/./03-create_devtest_set.sh /path_to_data
``` 

Then we have
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

To examine the elements in the dev and test TFRecord files, we can simply run the command line tool `view_tfrecord`

```bash
$ python3 -m neurst.cli.view_tfrecord /path_to_data/asr/devtest/DEV.tfrecords-00000-of-00001

features {
  feature {
    key: "audio"
    value {
      float_list {
        ......
        value: -0.5336110591888428
        value: -0.6020540595054626
        value: -0.5788766145706177
        value: -0.6893889307975769
      }
    }
  }
  feature {
    key: "src_lang"
    value {
      bytes_list {
        value: "en"
      }
    }
  }
  feature {
    key: "transcript"
    value {
      bytes_list {
        value: "in this week\'s episode, we drive the new ford ranger. we answer questions about e v\'s and driving in cold weather."
      }
    }
  }
  feature {
    key: "uuid"
    value {
      bytes_list {
        value: "724c4c23-a096-4a57-9aa9-be4a1e9af2ce"
      }
    }
  }
}

elements: {
    "transcript": bytes (str)
    "uuid": bytes (str)
    "audio": float32
    "src_lang": bytes (str)
}
```

## Training and evaluation

The training and evaluation procedures are the same as those of [AugmentedLibrispeech](/examples/speech_to_text/augmented_librispeech/README.md).

Specifically, if you are training the XL subset, please use the `speech_transformer_l` model. 

Also, please use the default arguments in `asr_training_args.yml` if possible, especially the `batch_size`. If you train a large model with a small `batch_size`, e.g. `speech_transformer_l` with batch_size=20,000, the training procedure is hard to converge.