# HistRoNER - Historical Romanian Named Entity Recognition
A benchmark comparing BiLSTM and fine-tuned BERT-based models for Named Entity Recognition on historical Romanian text from the period 1800–1950.


```mermaid
flowchart TD
    DS[(HistNERo dataset\navramandrei/histnero)] --> PRE[Preprocessing\nstrip punctuation · max_len=200]
    PRE --> VOC[Build vocab\nword → id]
    VOC --> EMB[nn.Embedding learned\nvocab_size × 300]
    PRE --> CAS[Casing features\n3-dim one-hot]
    EMB --> CAT[Concat · dim 303]
    CAS --> CAT
    CAT --> DROP1[Dropout 0.5]
    DROP1 --> LSTM[BiLSTM · 2 straturi\nhidden=256 · out 512]
    LSTM --> DROP2[Dropout 0.5]
    DROP2 --> FC1[Linear 512→256 + ReLU]
    FC1 --> FC2[Linear 256→11]
    FC2 --> LOSS[CrossEntropyLoss\nclass weights + label smoothing 0.1]
    LOSS --> OPT[AdamW lr=1e-3\nReduceLROnPlateau on val F1]
```
