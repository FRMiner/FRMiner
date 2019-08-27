# FRMiner

### Requirements
- [AllenNLP](https://github.com/allenai/allennlp)
- [PyTorch](https://github.com/pytorch/pytorch)
- [SpaCy](https://spacy.io/)


### File organization
- `data/`
    - `origin_data\`: original dialogues data 
    - `*_feature.txt\`: converted feature dialogues
    - `*_other.txt`: converted non-feature dialogues
- `src/`
    - `config.json`: a json file including settings
    - `finetune_config.json`: a json file for fine-tuning
    - `p_frminer_reader.py`: dataset reader for p-FRMiner
    - `p_frminer_model.py`: p-FRMiner model
    - `frminer_reader.py`: dataset reader for FRMiner
    - `frminer_model.py`: FRMiner model
    - `preprocess.py`: dataset preprocess and split
    - `siamese_metric.py`: metric for FRMiner
    - `util.py`: some util functions


### Configuration
`config.json` is the config file. Some key json fields in config file are specified as follows：

```json
"train_data_path": train file path
"validation_data_path": test file path
"text_field_embedder": word embedding, including pre-trained file and dimension of embedding 
"pos_tag_embedding": pos-tag embedding
"cuda_device": training with CPU or GPU
```

### Train

``allennlp train <config file> -s <serialization path> -f --include-package FRMiner``