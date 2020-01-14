# FRMiner

### Requirements
- Please check the [installation guide](INSTALL.md) to configure your environment



### File organization
- `data/`
    - `origin_data/`: original dialogues data 
    - `*_feature.txt`: converted feature dialogues
    - `*_other.txt`: converted non-feature dialogues
    - `glove.6B.50d.txt`: Pretrained word2vec file, and you need to download this file at [Glove](https://nlp.stanford.edu/projects/glove/), then put it into the folder
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


### Parameters Configuration
`config.json` is the config file. Some key json fields in config file are specified as follows：

```json
"train_data_path": train file path
"validation_data_path": test file path
"text_field_embedder": word embedding, including pre-trained file and dimension of embedding 
"pos_tag_embedding": pos-tag embedding
"cuda_device": training with CPU or GPU
```

### Train & Test

Open terminal in the parent folder which is the same directory level as `FRMiner` and run
``allennlp train <config file> -s <serialization path> -f --include-package FRMiner``.

For example, with `allennlp train FRMiner/config.json -s FRMiner/out/ -f --include-package FRMiner`, you can get
the output folder at `FRMiner/out` and log info showed on the console.

At the end of running process, the console outputs the final result for train and test with json format. The picture below is an output example:

