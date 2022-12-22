# **SimCSE Text Embeddings: Validation & Extension**

## Project Overview 

In this project, we analyze findings from the paper _SimCSE: Simple Contrastive Learning of Sentence Embeddings_ by Tianyu Gao, Xingcheng Yao, Danqi Chen of Princeton University and Tsinghua University. 

(Paper [here](https://arxiv.org/pdf/2104.08821.pdf), Codebase [here](https://github.com/princeton-nlp/SimCSE). Notably, this repository is forked from the original SimCSE repository. 

Specifically, we first attempt to replicate the findings of the paper. Then, we build on top of the embeddings published by the paper (SimCSE Embeddings) in order to conduct further validation. These validation components are represented as Extension 1 and Extension 2. Finally, we mention that we restrict this analyze to only the supervised SimCSE experiment from the aforementioned paper. 

## Table of Contents 

| Section of Project | Relevant Files                                                                                                                             |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Replication        | `evaluation.py`                                                                                                                            |
| Extension 1        | `project_extension/train_sentiment_classifier.py` `project_extension/neural_sentiment_classifier.py` `project_extension/tweet_dataset.py/` |
| Extension 2        | `project_extension/train_sentiment_classifier.py`   
| Final Paper        | `SimCSE_Analysis_Paper.pdf` [(link)](https://github.com/aks5bx/SimCSE/blob/main/SimCSE_Analysis_Paper.pdf)   



# Replication 

## Executive Summary

In short, we are able to almost perfectly replicate the results of the original paper. We do this by forking the original respository, re-connecting the original dataset, updating packages to reflect most recent versioning, and re-running the given evaluation scripts. Additional detail is included in the following sub-sections. 

## Setup

### Environment Setup 

(We choose Option 2 in order to mimic the original environment as best as possible, however, we also provide Option 1 for future users of our forked repository). 

**Option 1**
- Create and use a conda evironment from the yml file 

```
conda env create --name <INSERT NAME> --file=sim_env.yml
```

**Option 2**
- Create empty conda environment and install packages 
- Install required pacakges using `pip install -r requirements.txt`

### Running the Code 

#### Pre-run Setup 
- Make sure to build data using the sh scripts in the data/ and SentEval/data directories
- Results will be written to the files: `eval_results.txt`, `train_results.txt`, and `output.txt`

In order to generate test output (in the same format as in the original SimCSE repo, we run the following command: 

```
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode test
```

## Comparing Results 

Note: the results of the Princeton SimCSE runs have been gathered from their published paper (linked above). Specifically, we report the results of their runs via Table Five on Page Seven. 

The runs we are interested in are in the Supervised section of the table. The SimCSE runs are denoted in the table using the ∗ symbol. For each of the embedding initializations (BERT Base, RoBERTa Base, RoBERTa Large), we report our results and those from the original paper. 

### BERT Base Case

**Our Run**
```

------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 75.30 | 84.67 | 80.19 | 85.40 | 80.82 |    84.26     |      80.39      | 81.58 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

**Princeton SimSCE Run**
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 75.30 | 84.67 | 80.19 | 85.40 | 80.82 |    84.26     |      80.39      | 81.58 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

### RoBERTa Base Case 

_The only differences here appear to be from differences in rounding._

**Our Run**
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 76.53 | 85.20 | 80.95 | 86.03 | 82.56 |    85.83     |      80.50      | 82.51 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

**Princeton SimSCE Run** 
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 76.53 | 85.21 | 80.95 | 86.03 | 82.57 |    85.83     |      80.50      | 82.52 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

### RoBERTa Large Case 

_The only differences here appear to be from differences in rounding._

**Our Run**
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 77.45 | 87.27 | 82.36 | 86.66 | 83.93 |    86.70     |      81.95      | 83.76 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

**Princeton SimSCE Run** 
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 77.46 | 87.27 | 82.36 | 86.66 | 83.93 |    86.70     |      81.95      | 83.76 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

# Extentions

## Executive Summary 

As noted, we develop two extensions to further validate the results from the SimCSE paper. We briefly describe them here, but in-depth explanations of both of the extensions can be found in our final paper. 

### Extension 1: Training and Comparing BERT Model vs SimCSE Model on Sentiment Prediction Task

In this extension, we use sentence encoders to encode tweets in tw_sentiment_df, freeze embeddings, then input them into a simple feed-forward neural networks to predict sentiment classes (positive, neutral, or negative). With respect to the sentence encoder, we create two feed-forward networks, one using BERT embeddings and another using SimCSE embeddings. We then compare the two networks based on accuracy on our downstream sentiment prediction task. 

Note that we use a weakly supervised training regime, as labels were generated by generating polarity scores on each tweet using TextBlob and binning them into positive, negative, and neutral sentiment classes.

### Extension 2: Leveraging Feature Permutation to Compare BERT Embeddings and SimCSE Embeddings

Here, we use a similar approach as the one outlined for Extension 1. However, instead of using two separate networks and comparing performance on a downstream task, we use a single network. The key difference here is that while Extension 1 featured a network with _either_ SimCSE or BERT Embeddings, Extension 2 uses a network with _both_ SimCSE and BERT Embeddings. As in, we concatenate both embeddings next to each other and provide the concatenated embeddings as input to our one network. We then train our network on the same downstream sentiment prediction task using both sets of embeddings. 

During validation, we conduct feature permutation where we randomize either the SimCSE or the BERT embeddings. The intuition here is that we begin by giving both the unpermuted SimCSE and BERT embeddings to our network. Then, we allow our network to learn sentiment classification using any combination of SimCSE and/or BERT features. If the network finds more value in using one embedding type (SimCSE or BERT) over another, then when we permute that embedding type in our validation, we should see a larger dropoff in accuracy.

### Data Generation & EDA: 

_Note: This has already been run and the dataset is available in the repo_

We take the following steps, in order, to generate the data required for both of our extensions: 

1. We run the file `generate_tweet_data.py "fifa" <subset> <num datapoints>`  
- (fifa) is the query arg
- if subset arg is `'subset'` then we will only take tweets that contain references to countries in the world cup groups B, D, G (see note below)

2. Locate data output in the file `sentiment_data/tw_sentiment_df.csv` (with an extension that signifies the dataset size as _XX)

3. Limited EDA is available in `data_analysis/sentiment_data_exp.py`

_Note: we subset to these countries in order to stay under the query length limits stipulated by the Twitter API. The groups are otherwise arbitrarily chosen._ 

### Model Components

- `train_sentiment_classifier.py` - trains neural sentiment classifier
- `neural_sentiment_classifier.py` - defines sentiment classifier model
- `tweet_dataset.py` - defines torch dataset object and collate function for batching

### Training Model

To train model, we run:

```python train_sentiment_classifier.py [model]```

where model = 'bert', 'simcse', or 'both'. Choosing 'both' generates bert and simcse encodings of input sentences and concatenate them before
feeding into neural sentiment classifier. Choosing 'bert' or 'simcse' generates encodings of the specified type. 

#### Additional CLI Options
Additional options for running train_sentiment_classifier.py:

- `--path_to_data` tw_sentiment_df.csv by default, specifies path to twitter data (must be csv with 'tweet' and 'sentiment' columns)
- `--tune` tunes model with 45 different hparam configurations
- `--save` saves model state dict at the end of training as `[model]_final.pt`
- Train the classifier using `train_sentiment_classifier.py`

## Results 

The following results are paraphrased excerpts from our final paper: 

### Extension 1

After hyperparameter tuning, we compared sentiment classification accuracy on our final holdout set at the respective optimal configurations for BERT and SimCSE. To ensure our results were robust, we computed mean test accuracy across 6 replicates per method, where in each replicate we generated random 70% train / 30% test splits, trained SimCSE and BERT feedforward networks using their optimal hyperparameters, and computed their respective test accuracies. We also conducted these experiments over a range of dataset sizes from 1,000 to 10,000 to determine the sensitivity of each sentiment classifier model to training size. 

Our results highlight two key advantages our SimCSE sentiment classifier has over the baseline BERT model. The first is that we achieve higher maximum mean test accuracy with SimCSE compared to BERT. Our SimCSE sentiment classifier achieves a maximum mean test accuracy of 0.88, while its BERT-only counterpart achieves 0.74. 

The second advantage is that SimCSE achieves close to maximum performance with less training time and training data than BERT. We can see that the SimCSE sentiment classification model reaches near-maximum performance between epochs 5 and 10 and a dataset size of roughly 4,000, while the BERT classification model validation performance does not begin to level off until epoch 15 and a dataset size of 6,000.

**In short, we are able to show marked improvement on our downstream sentiment classification task when using SimCSE Embeddings compared to Base BERT Embeddings.**

### Extension 2

We found that after permuting the BERT embeddings, the model attained a mean test accuracy of 0.73, whereas when permuting the SimCSE embeddings, the model attained a mean test accuracy of 0.48. What this tells us is that the model likely ”learned” more from the SimCSE embeddings than the BERT embeddings. This aligns with findings in our first experiment and explains why our model performance decreased the most when we stripped the interpretability and meaning from the SimCSE embeddings.

**In short, we are able to show that when our model receieves both BERT and SimCSE Embeddings, it exhibits a higher reliance on the SimCSE Embeddings in order to complete our downstream sentiment classification task.**

### Summary 

In sum, our results demonstrate how leveraging a contrastive learning framework such as SimCSE can improve sentence embeddings and ultimately improve performance in downstream NLP tasks such as sentiment analysis.

## Technology Used 

We make special note of the following technologies used for this project: 
- Python 
- PyTorch 
- Twitter API 
- Huggingface
- BERT/roBERTa

## Final Remarks 

### Contributions 
- Jin Ishizuka: Wrote script to pull tweet data via the Twitter API and generate target sentiment labelings using the TextBlob library 
- Andre Chen: Created tweet preprocessing pipeline and built, tuned, and evaluated senti- ment classifier neural net model for BERT and SimCSE embeddings
- Aditya Srikanth: Generated dataset using API script and tweet pipeline. Conducted BERT and SimCSE embeddings permutation, trained network on permuted inputs and collected results

### References 

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. Simcse: Simple contrastive learning of sentence em-beddings.

Nikita Silaparasetty. 2022. [Twitter sentiment analysis for data science using python in 2022](https://medium.com/@nikitasilaparasetty/twitter-sentiment-analysis-for-data-science-using-python-in-2022-6d5e43f6fa6e).
