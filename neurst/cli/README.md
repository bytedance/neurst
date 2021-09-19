## CLI

NeurST provides various command line tools.

### Contents

* [`run_exp`: Train and evaluate](#train-and-evaluate)
* [`view_registry`: List registered classes and arguments](#list-registered-classes-and-arguments)
* [`view_tfrecord`: Examine a TFRecord file](#examine-a-tfrecord-file)
* [`inspect_checkpoint`: Inspect variables in a checkpoint](#inspect-variables-in-a-checkpoint)
* [`convert_checkpoint`: Convert a checkpoint from a well-trained model](#convert-a-checkpoint-from-a-well-trained-model)
* [`audio_tfrecord_analysis`: Analysis audio TFRecord dataset](#analysis-audio-tfrecord-dataset)

------

### Train and evaluate
See [examples/](/examples)

### List registered classes and arguments
This tool can list all registered classes (`Model`, `Task`, ...), their arguments and detailed explanation:    

- to see user manual:
```bash
$ python3 -m neurst.cli.view_registry

Usage: 
    >> python3 -m neurst.cli.view_registry registry_name
           Show registered classes and their aliases.

    >> python3 -m neurst.cli.view_registry registry_name class_name
           Show detailed parameters of the class.

All registry names: 
    - criterion
    - metric
    - feature_extractor
    - data_pipeline
    - tokenizer
    - dataset
    - entry
    - base_layer
    - search_method
    - model
    - decoder
    - encoder
    - hparams_set
    - validator
    - optimizer
    - lr_schedule
    - task
```

- to list the registered tasks:
```bash
$ python3 -m neurst.cli.view_registry task

All registered task(s): 
    |  Class  |  Aliases  |
    |  Seq2Seq  |  seq2_seq, Seq2Seq, seq2seq, seq_to_seq  |
    |  AudioToText  |  audio_to_text, audio2text, AudioToText, audiototext  |
```

- to see the detailed explanation of arguments for `Seq2Seq`:
```bash
$ python3 -m neurst.cli.view_registry task Seq2Seq

Flags for Seq2Seq:
    |  flag  |  type  |  default  |  help  |
    |  shuffle_buffer  |  <class 'int'>  |  0  |  The buffer size for dataset shuffle.  |
    |  batch_size  |  <class 'int'>  |  None  |  The number of samples per update.  |
    |  batch_size_per_gpu  |  <class 'int'>  |  None  |  The per-GPU batch size, that takes precedence of `batch_size`.  |
    |  cache_dataset  |  <class 'bool'>  |  None  |  Whether to cache the training data in memory.  |
    |  max_src_len  |  <class 'int'>  |  None  |  The maximum source length of training data.  |
    |  max_trg_len  |  <class 'int'>  |  None  |  The maximum target length of training data.  |
    |  truncate_src  |  <class 'bool'>  |  None  |  Whether to truncate source to max_src_len.  |
    |  truncate_trg  |  <class 'bool'>  |  None  |  Whether to truncate target to max_trg_len.  |
    |  batch_by_tokens  |  <class 'bool'>  |  None  |  Whether to batch the data by word tokens.  |
Dependent modules for Seq2Seq: 
    |  name  |  module  |  help  |
    |  src_data_pipeline  |  data_pipeline  |  The source side data pipeline.  |
    |  trg_data_pipeline  |  data_pipeline  |  The target side data pipeline.  |
```

### Examine a TFRecord file
`view_tfrecord` can examine a TFRecord file by displaying one data sample and the element types.
```bash
$ python3 -m neurst.cli.view_tfrecord tests/examples/train-zh2en-tfrecords-00000-of-00004

features {
  feature {
    key: "feature"
    value {
      int64_list {
        value: 27
        value: 146
        ......
      }
    }
  }
  feature {
    key: "label"
    value {
      int64_list {
        value: 38
        value: 29
        ...
      }
    }
  }
}

elements: {
    "label": int64
    "feature": int64
}
``` 

### Inspect variables in a checkpoint
`inspect_checkpoint` can list all variable names and shapes in a TensorFLow checkpoint. 

- to see the user manual:
```bash
$ python3 -m neurst.cli.inspect_checkpoint 

Usage: 
    >> python3 -m neurst.cli.inspect_checkpoint modeldir_or_checkpoint (--structured)
           List all variables and their shapes.

    >> python3 -m neurst.cli.inspect_checkpoint model_dir/checkpoint regular_expr
           List the variables and their shapes if the name matches the `regular_expr`.

    >> python3 -m neurst.cli.inspect_checkpoint model_dir/checkpoint var_name
           Print the variable tensor.
```

- to list all variables and shapes (a tiny transformer model):
```bash
$ python3 -m neurst.cli.inspect_checkpoint ./test_models

        variable name    shape
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/encdec_attention/kv_transform/bias       [16]
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/encdec_attention/kv_transform/kernel     [8, 16]
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/encdec_attention/output_transform/bias   [8]
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/encdec_attention/output_transform/kernel [8, 8]
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/encdec_attention/q_transform/bias     
......
```

- to list the specific variables (e.g. layer norm):
```bash
$ python3 -m neurst.cli.inspect_checkpoint ./test_models "/ln/"

        variable name (/ln/)     shape
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/ln/beta  [8]
SequenceToSequence/TransformerDecoder/layer_0/encdec_attention_prepost_wrapper/ln/gamma [8]
SequenceToSequence/TransformerDecoder/layer_0/ffn_prepost_wrapper/ln/beta       [8]
SequenceToSequence/TransformerDecoder/layer_0/ffn_prepost_wrapper/ln/gamma      [8]
......
```

- to print the tensor (e.g. target embedding table)
```bash
$ python3 -m neurst.cli.inspect_checkpoint ./test_models "SequenceToSequence/target_symbol_modality_posenc_wrapper/target_symbol_modality/shared/weights"

Variable name: SequenceToSequence/target_symbol_modality_posenc_wrapper/target_symbol_modality/shared/weights
Tensor Shape: [164, 8]
Tensor Value: 
[[-0.17575996  0.1493602  -0.18946414 ... -0.31306902  0.08493289
   0.23959549]
 [ 0.90474224 -0.609249    0.03951814 ...  0.35367873 -0.20142521
  -0.52413136]
 ...
 [-0.23102537 -0.21176471  0.00733235 ... -0.4419487   0.02791528
   0.37591913]]
```

### Convert a checkpoint from a well-trained model
`convert_checkpoint` can convert a well-trained model to neurst's checkpoint.

- to see the supported model names:
```bash
$ python3 -m neurst.cli.view_registry converter

All registered converter(s): 
    |  Class  |  Aliases  |
    |  GoogleBert  |  googlebert, GoogleBert, google_bert  |
```

- to convert a publicly available `bert-base-uncased` model:
```bash
$ python3 -m neurst.cli.convert_checkpoint --model_name google_bert --from bert-base-uncased --to /path/to/save
```
or to convert a fairseq transformer model:
```bash
$ python3 -m neurst.cli.convert_checkpoint --model_name fairseq_transformer --from /path/to/transformer.pt --to /path/to/save
```

### Analysis audio TFRecord dataset
It will print the metadata of the audio dataset, i.e. the number of samples and the duration of all audio segments:
```bash
python3 -m neurst.cli.analysis.audio_tfrecord_analysis --data_path ... --feature_extractor fbank/float_identity ...
```
