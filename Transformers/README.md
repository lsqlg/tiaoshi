## Transformer Models - BERT and DistilBERT 

This folder contains are implementation of pre-trained BERT and DistilBERT models from Transformers (Hugging Face) customized multi-label text classification.

此文件夹包含来自变形金刚（拥抱脸）定制的多标签文本分类的预训练BERT和DistilBERT模型的实现

## Get up and running

1. Make sure that the datasets are placed into a subfolder `./data/multi_label_data`.

2. Check for dependencies `numpy`, `torch`, `transformers`, `pandas`,`sklearn`,`tqdm` and `matplotlib`

确保将数据集放在子文件夹中。/data/multi_label_data。
检查依赖关系numpy、torch、transformers、panda、sklearn、tqdm和matplotlib
## Code overview

- In `bert_model_multi_label.py`and `distilbert_model_multi_label.py` you can find the implementation for the BERT and DistilBERT models for multi label classification.
  The experimental setup can be configured in the beginning of the script:
    - valid datasets are: {'amazon', 'dbpedia', 'econbiz', 'nyt', 'reuters', 'rcv1-v2', 'goemotions'}
    - the corresponding label number can be found in the python files.

代码概述
在bert_model_multi_label中。py和distilbert_model_multi_label。py可以找到用于多标签分类的BERT和DistilBERT模型的实现。可以在脚本的开头配置实验设置：
有效的数据集为：｛'amazon'、'dbpedia'、'ecobiz'、'nyt'、'reuters'、'rcv1-v2'、'goemotions'｝
相应的标签号可以在python文件中找到

## Running experiments

The scripts `bert_model_multi_label.py` and `distilbert_model_multi_label.py` are the main entry point for running an experiment. Variables at the beginning of the scripts should be set accordingly to datasets and label number.
运行实验
脚本bert_model_multi_label。py和distilbert_model_multi_label。py是运行实验的主要入口点。脚本开头的变量应根据数据集和标签号进行相应设置。
## References

    @inproceedings{wolf-etal-2020-transformers,
    title = {Transformers: State-of-the-Art Natural Language Processing},
    author = {Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    month = {oct},
    year = {2020},
    address = {Online},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6},
    pages = {38--45}
    }

