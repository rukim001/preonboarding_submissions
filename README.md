<h1 align="center">Semantic Textural Similarity 👋</h1>

<p align="center">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/boostcampaitech2/klue-level2-nlp-03?style=social">
  <img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/boostcampaitech2/klue-level2-nlp-03?style=plastic">
  <img alt="Conda" src="https://img.shields.io/conda/pn/boostcampaitech2/klue-level2-nlp-03">
</p>  

## Overview Description

Semantic Textual Similarity (STS) is to measure the degree of semantic equivalence between two sentences. We include KLUE-STS in our benchmark because it is essential to other NLP tasks such as machine translation, summarization, and question answering. Like STS in GLUE, many NLU benchmarks include comparing semantic similarity of text snippets such as semantic similarity, paraphrase detection, or word sense disambiguation.


We formulate STS as a sentence pair regression task which predicts the semantic similarity of two input sentences as a real value from 0 (no meaning overlap) to 5 (meaning equivalence). A model performance is measured by Pearson's correlation coefficient. We additionally binarize the real numbers into two classes with a threshold score 3.0 (paraphrased or not), and use F1 score to evaluate the model.


## Evaluation Methods
The evaluation metrics for KLUE-STS is 1) Pearson's correlation coefficient (Pearson' r), and 2) F1 score.


Pearson's r is a measure of linear correlation between human-labeled sentence similarity scores (ground truths) and model predicted scores, adopted in STS-b \cite{cer-etal-2017-semeval}. Since our dev/test set scores are balanced, the coefficient correctly gives the magnitude of the relationship.


F1 is adopted to measure binarized results (paraphrased or not), which is defined as the harmonic mean of precision and recall, where precision is the ratio of true positives to all predicted positives and recall is the ratio of true positives to all actual positives. Specifically, we use `sklearn.metrics.f1_score` with 'binary' averaging, which means we report results for the class specified by ""paraphrased"". Our F1 score weights recall and precision equally to incentivize a model which maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.





## Code Contributors

<p>
<a href="https://github.com/jiho-kang" target="_blank">
  <img x="5" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/43432539?v=4"/>
</a>
<a href="https://github.com/tjddn5242" target="_blank">
  <img x="74" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/61862332?v=4"/>
</a>
<a href="https://github.com/rukim001" target="_blank">
  <img x="143" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/92706101?v=4"/>
</a>
<a href="https://github.com/sw6820" target="_blank">
  <img x="212" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/52646313?v=4"/>
</a>
<a href="https://github.com/yjinheon" target="_blank">
  <img x="281" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/37974827?v=4"/>
</a>
<a href="https://github.com/seawavve" target="_blank">
  <img x="350" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/66352658?v=4"/>
</a>

</p>

## Environments 

### OS
<!--  - UBUNTU 18.04 -->

### Requirements
```
datasets==1.5.0
transformers==4.5.0
tqdm==4.41.1
pandas==1.1.4
scikit-learn==0.24.1
konlpy==0.5.2
numpy==1.21.3
faiss-gpu==1.7.1.post2
rank_bm25==0.2.1
pororo==0.4.2
```
### Hardware
The following specs were used to create the original solution.
<!-- - GPU(CUDA) : v100  -->

## Reproducing Submission
<!-- To reproduct my submission without retraining, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Prepare Datasets](#Prepare-Datasets)
4. [Download Baseline Codes](#Download-Baseline-Codes)
5. [Train models](#Train-models-(GPU-needed))
6. [Inference & make submission](#Inference-&-make-submission)
7. [Ensemble](#Ensemble)
8. [Wandb graphs](#Wandb-graphs) -->

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
$ bash ./install/install_requirements.sh
```

## Dataset Preparation
All json files are already in data directory.
```
# data (51.2 MB)
tar -xzf data.tar.gz
```
### Prepare Datasets
After downloading  and converting datasets and baseline codes, the data directory is structured as:
```
├── code
│   ├── assets
│   │    ├── system_assets1.png
│   │    ├── system_assets2.png
│   │    ├── train_assets.png
│   │    └── dataset.png
│   ├── install
│   │    └── install_requirements.sh
│   ├── ensemble_csv
│   │    ├── ensemble.ipynb
│   │    ├── klue-bert-base__BM5_topk_8.csv
│   │    ├── klue-bert-base__dpr_train_topk_5.csv
│   │    ├── koelectra-base__BM25_topk_5.csv
│   │    ├── roberta_cnn__batch_16__BM5_topk_5.csv
│   │    └── roberta_cnn__batch_8__BM25_topk_5.csv
│   ├── arguments.py
│   ├── bm25_retrieval.py
│   ├── inference.py
│   ├── inference_command.txt
│   ├── model.py
│   ├── README.md
│   ├── README_KO.md
│   ├── retrieval.py
│   ├── retrieval_inference.py
│   ├── run.sh
│   ├── train.py
│   ├── train_command.txt
│   ├── trainer_qa.py
│   ├── utils_qa.py
│   └── wiki_preprocess.py
└── data
    ├── test_dataset
    │    ├── dataset_dict.json
    │    └── validataion
    │          ├── dataset.arrow
    │          ├── dataset_info.json
    │          ├── indices.arrow
    │          └── state.json
    ├── train_dataset
    │          ├── train    
    │          │    ├── dataset.arrow
    │          │    ├── dataset_info.json
    │          │    ├── indices.arrow
    │          │    └── state.json
    │          ├── validation
    │          │    ├── dataset.arrow
    │          │    ├── dataset_info.json
    │          │    ├── indices.arrow
    │          │    └── state.json    
    │          └── dataset_dict.json
    └── wikipedia_documents.json

```
#### Download Baseline code
To download baseline codes, run following command. The baseline codes will be located in `/opt/ml/code`
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/code.tar.gz
```

#### Download Dataset
To download dataset, run following command. The dataset will be located in `/opt/ml/dataset`
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz
``` 
### Train Models (GPU needed)
#### Extractive Model
To train extractive models, run following commands.
```
$ python train.py --output_dir ./models/train_dataset --do_train
```
#### Generative Model
To train generative models, run following commands.
```
$ python generation.py --output_dir ./models/train_dataset --do_train
```
<!-- 
The expected training times are:

Model | GPUs | Batch Size | Training Epochs | Training Time
------------  | ------------- | ------------- | ------------- | -------------
 roberta-large + cnn | v100 | 16 | 3 | 34m 18s
 bart-base | v100 | 8 | 3 | 11m 58s
 bert-base | v100 | 16 | 5 | 25m 07s 
 koelectra-base | v100 | 16 | 3 | 15m 43s
 t-base | v100 | 8 | 3 | 9m 57s
 -->

### Inference & Make Submission
```
$ python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### Wandb Graphs
<!-- - Train Graphs
<p>
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/train_assets.PNG">
</p>    

- System Graphs
<p>
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/system_assets1.PNG">
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/system_assets2.PNG">
</p> -->

## Reference
[KLUE-MRC - Machine Reading Comprehension](https://klue-benchmark.com/tasks/72/data/description)
