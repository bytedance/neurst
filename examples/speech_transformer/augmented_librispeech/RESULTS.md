# Results on Argumented LibriSpeech


### Comparison with counterparts (speech_transformer_s)
test, case-insensitive

|Model|tok|detok|
|---|---|---|
|Transformer ST + ASR PT (1)| - |15.5|
|Transformer ST + ASR/MT PT (1)| - |16.2|
|Transformer ST + ASR/MT PT + SpecAug (1) | - |16.7|
|Transformer ST ensemble 3 models (1) | - | 17.4|
|Transformer ST + ASR/MT PT (2)| 14.3 | - |
|Transformer ST + ASR/MT PT + KD (2) | 17.0 | - |
|Transformer ST + ASR PT + SpecAug (3) | 16.9 | - |
|Transformer ST + ASR PT + curriculum pre-training + SpecAug (3) | 18.0 | - |
|Transformer ST + ASR PT (4) | 15.3 | - |
|Transformer ST + triple supervision (TED) (4) | 18.3 | - |
|**NeurST** Transformer ST + ASR PT | 17.9 | 16.5 |
|**NeurST** Transformer ST + ASR PT + SpecAug | 18.7 | 17.2 |
|**NeurST** Transformer ST ensemble 2 models | **19.2** | **17.7**|

(1) Espnet-ST (Inaguma et al., 2020) with additional techniques: speed perturbation, pre-trained MT decoder and CTC loss for ASR pretrain;

(2) Liu et al. (2019) with the proposed knowledge distillation;

(3) Wang et al. (2020) with additional ASR corpora and curriculum pre-training;

(4) Dong et al. (2020) with CTC loss and a pre-trained BERT encoder as supervision with external ASR data;


### ASR (dmodel=256, WER) 

|Framework|Model|Dev|Test| |
|---|---|---|---|---|
|NeurST|Transformer ASR |8.8|8.8| pure end-to-end, beam=4, no length penalty |
|Espnet (Inaguma et al., 2020)| Transformer ASR + ctc | 6.5 | 6.4 | multi-task training with ctc loss | 


### MT and ST (dmodel=256, case-sensitive, tokenized BLEU/detokenized BLEU)

|Framework|Model|Dev|Test|
|---|---|---|---|
|NeurST|Transformer MT |20.8 / 19.3 | 19.3 / 17.6 |
|NeurST|cascade ST (Transformer ASR -> Transformer MT) | 18.3 / 17.0| 17.4 / 16.0 |
|NeurST|end2end Transformer ST + ASR pretrain | 18.3 / 16.9 | 16.9 / 15.5  |
|NeurST|end2end Transformer ST + ASR pretrain + SpecAug | 19.3 / 17.8 | 17.8 / 16.3  |
|NeurST|end2end Transformer ST ensemble above 2 models | 19.3 / 18.0 | 18.3 / 16.8  |

### MT and Cascade ST (dmodel=256, case-insensitive, tokenized BLEU/detokenized BLEU)

|Framework|Model|Dev|Test|
|---|---|---|---|
|NeurST|Transformer MT | 21.7 / 20.2 | 20.2 / 18.5 |
|Espnet (Inaguma et al., 2020)| Transformer MT| ---- / 19.6 | ---- / 18.1 |
|NeurST|cascade ST (Transformer ASR -> Transformer MT) | 19.2 / 17.8 | 18.2 / 16.8 |
|Espnet (Inaguma et al., 2020)| cascade ST (Transformer ASR + ctc -> Transformer MT) | ---- / ---- | ---- / 17.0 |



