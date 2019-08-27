# FRMiner

### Requirements
- [AllenNLP](https://github.com/allenai/allennlp)
- [PyTorch](https://github.com/pytorch/pytorch)
- [SpaCy](https://spacy.io/)


### File organization
- `config.json`: a json file including settings
- `finetune_config.json`: a json file for fine-tuning
- `dialog_reader.py`: dataset reader for plain FRMiner
- `dialog_model.py`: plain FRMiner model
- `dialog_reader_siamese.py`: dataset reader for Siamese FRMiner
- `model.py`: Siamese model
- `preprocess.py`: dataset preprocess and split
- `siamese_metric.py`: metric for Siamese FRMiner
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