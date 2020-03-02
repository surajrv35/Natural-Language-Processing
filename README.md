## RNN-Encoder-Decoder
PyTorch implementation of recurrent neural network encoder-decoder architecture model 
for statistical machine translation, as detailed in this paper:
[Learning Phrase Representations using RNN Encoder–Decoder 
for Statistical Machine Translation (Cho et al., 2014)](https://arxiv.org/pdf/1406.1078.pdf)

### Dataset
WMT'14 workshop [translation task](http://www.statmt.org/wmt14/translation-task.html).

### Imporved Model

The models directory contains three models -- 
1. baseline.model
2. encoder.ckpt
3. decoder.ckpt

So, the enhancements have been added in them. The number of layers in the neural network have been changed to 5 from 1. We forgot to change the name of the baseline.model to improved.model.

### The test for the Project

1. The data directory contains 2 files testDataEnglishTxt and testDataFrenchTxt which contains the sentences that are going to be tested.

