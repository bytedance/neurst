# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Instruction for 
    - CTNMT (Yang et al., 2020) training
    - Prune-Tune (Liang et al., 2021)
- dataset for IWSLT offline ST task 
- language model task and GPT-2 pretraining


### Changed


### Fixed


## [0.1.1] - 28th March, 2021
### Added
- PyTorch version Transformer & SpeechTransformer model.
- Audio extraction for CommonVoice/IWSLT.
- Data sampler and dataset for multilingual machine translation
- Mixed training dataset with data sampler.
- Multilingual Translation task
- Instruction for 
    - training transformer models on WMT14 EN->DE
    - weight pruning 
    - quantization aware training for transformer model

### Fixed
- Compat with TensorFlow v2.4



## [0.1.0] - 25th Dec., 2020
### Added
- Basic code structure for Encoder, Decoder, Model, DataPipeline, Tokenizer, Experiment, Metric, and Dataset.
- (Model) Adds implementation of pre-norm/post-norm Transformer, Speech Transformer, BERT, GPT-2, and Wav2Vec2.0.
- (Task) Adds implementation of sequence to sequence task and speech to text task (ASR, ST).
- (DataPipeline, Tokenizer) Adds wrappers for commonly used tokenizers: moses, bpe, jieba, character, sentencepiece, etc.
- (Dataset) Adds support for reading parallel corpus, speech corpora (libri-trans, MuST-C, and LibriSpeech), and TFRecords.
- (Experiment) Adds implementation of common training procedure with mixed precision training and various distributed strategies (`MirroredStrategy`, `Horovod`, `Byteps`).
- (Metric) Adds implementation of BLEU and WER metrics.
- (Converter) Adds implementation of converting checkpoints from google BERT, OpenAI GPT-2, fairseq Transformer, and fairseq Wav2Vec2.0.
- Add support for converting checkpoints from publicly 
- Beam search decoding and top-k/p sampling.
- Supports averaging checkpoints, TFRecord generation, model restoring (see [cli/README.md](/neurst/cli/README.md)).
- Step-by-step recipes for training an end-to-end speech translation model (see [examples/speech_to_text](/examples/speech_transformer)).

