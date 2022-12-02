# **NLP Team II Project Repo**

TABLE OF CONTENTS
- Part I: Replication 
- Part II: Extensions

# Part I: Replication 

## Setup

#### Environment Setup 

**Option 1**
- Create and use a conda evironment from the yml file 

```
conda env create --name <INSERT NAME> --file=sim_env.yml
```

**Option 2**
- Create your own conda environment and install packages 
- Installed required pacakges using `pip install -r requirements.txt`

### Running the Code 

- Make sure to build data using the sh scripts in the data/ and SentEval/data directories
- Results are in the eval_results.txt, train_results.txt files, output.txt


## Comparative Runs

Note: the results of the Princeton SimCSE runs have been gathered from their published paper. Specifically, we report the results of their runs via Table 5 on Page 7. 

The runs we are interested in are in the Supervised section of the table. The SimCSE runs are denoted in the table using the ∗ symbol. 

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

## Debug Trial

Train Results 
```
epoch = 1.0
train_runtime = 748.7578
train_samples_per_second = 2.877
```

Eval Results 
```
epoch = 1.0
eval_CR = 36.24
eval_MPQA = 68.77
eval_MR = 50.0
eval_MRPC = 32.46
eval_SST2 = 49.08
eval_SUBJ = 50.0
eval_TREC = 1.58
eval_avg_sts = nan
eval_avg_transfer = 41.16142857142857
eval_sickr_spearman = nan
eval_stsb_spearman = nan
```

# Part II: Extentions

## Setup

#### Environment Setup

**Option 1**
- Create and use a conda evironment from the yml file 

```
conda env create --name <INSERT NAME> --file=sim_env.yml
```

**Option 2**
- Create your own conda environment and install packages 
- Installed required pacakges using `pip install -r requirements.txt`

#### Data Generation & EDA: 

_Note: This has already been run and the dataset is available in the repo_

1. Run the file `generate_tweet_data.py "fifa" <subset> <num datapoints>`  
- (fifa) is the query arg
- if subset arg is `'subset'` then we will only take tweets that contain countries in the world cup (groups B, D, G)
2. Locate data output in the file `sentiment_data/tw_sentiment_df.csv` (with an extension that signifies the dataset size as _XX)
3. Limited EDA is available in `data_analysis/sentiment_data_exp.py`

## Extension I: Comparing BERT Model vs SimCSE Model 

- Train the classifier using `train_sentiment_classifier.py`

## Extension II: Feature Permutation 

- Under construction 

## Extension III (if needed): Sentiment Analysis 

- We could do sentiment analysis by country..if we want