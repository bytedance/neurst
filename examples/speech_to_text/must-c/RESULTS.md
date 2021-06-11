# Results on MuST-C


### Comparison with counterparts (speech_transformer_s)
test-COMMON, case-sensitive, detokenized BLEU

|Model|DE|ES|FR|IT|NL|PT|RO|RU|avg.|
|---|---|---|---|---|---|---|---|---|---|
|Transformer ST + ASR PT (1) | 21.8 | 26.4 | 31.6 | 21.5 | 25.2 | 26.8 | 20.5 | 14.3 | 23.5 |
| Transformer ST + ASR/MT PT (1) | 22.3 | 27.8 | 31.5 | 22.8 | 26.9 | 27.3 | 20.9 | 15.3 | 24.4| \\
| Transformer ST + ASR/MT PT + SpecAug (1) | **22.9** | **28.0** | 32.8 | **23.8** | **27.4** | 28.0 | 21.9 | **15.8** | **25.1** |
| Transformer ST + ASR PT + SpecAug (2) | 22.7 | 27.2 | 32.9 | 22.7 | 27.3 | 28.1 | 21.9 | 15.3 | 24.8|
| Transformer ST + adaptive feature selection (3) | 22.4 | 26.9 | 31.6 | 23.0 | 24.9 | 26.3 | 21.0 | 14.7 | 23.9|
|**NeurST** Transformer ST + ASR PT | 21.9 | 26.8 | 32.3 | 22.2 | 26.4 | 27.6 | 20.9 | 15.2 | 24.2|
|**NeurST** Transformer ST + ASR PT + SpecAug | 22.8 | 27.4 | **33.3**  | 22.9 | 27.2 | **28.7** | **22.2** | 15.1 | 24.9|


(1) Espnet-ST (Inaguma et al., 2020) with additional techniques: speed perturbation, pre-trained MT decoder and CTC loss for ASR pretrain;

(2) fairseq-ST (Wang et al., 2020) with the same setting as NeurST;

(3) Zhang et al. (2020) with the proposed adaptive feature selection method

### ASR (dmodel=256, WER) 
test-COMMON

|Framework|Model|DE|ES|FR|IT|NL|PT|RO|RU|
|---|---|---|---|---|---|---|---|---|---|
|NeurST|Transformer ASR |13.6|13|12.9|13.5|13.8|14.4|13.7|13.4|
|Espnet (Inaguma et al., 2020)| Transformer ASR + ctc |12.7|12.1|12|12.4|12.1|13.4|12.6|12.3|
|fairseq-ST (Wang et al., 2020)| Transformer ASR|18.2|17.7|17.2|17.9|17.6|19.1|18.1|17.7|


### MT and ST (dmodel=256, case-sensitive, tokenized BLEU/detokenized BLEU)
test-COMMON

|Framework|Model|DE|ES|FR|IT|NL|PT|RO|RU|
|---|---|---|---|---|---|---|---|---|---|
|NeurST|Transformer MT |27.9/27.8|32.9/32.8|42.2/40.2|29.0/28.5|32.9/32.7|34.4/34.0|27.5/26.4|19.3/19.1|
|NeurST|cascade ST (Transformer ASR -> Transformer MT) |23.5/23.4|28.1/28.0|35.8/33.9|24.3/23.8|27.3/27.1|28.6/28.3|23.3/22.2|16.2/16.0|
|NeurST|end2end Transformer ST + ASR pretrain |21.9/21.9|26.9/26.8|34.2/32.3|22.6/22.2|26.5/26.4|27.8/27.6|21.9/20.9|15.0/15.2|
|NeurST|end2end Transformer ST + ASR pretrain + SpecAug |22.8/22.8|27.5/27.4|35.2/33.3|23.4/22.9|27.4/27.2|29.0/28.7|23.2/22.2|15.2/15.1|

