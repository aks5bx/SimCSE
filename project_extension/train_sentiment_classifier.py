from argparse import ArgumentParser
from functools import partial
import itertools
import os
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from neural_sentiment_classifier import NeuralSentimentClassifier
from tweet_dataset import TweetDataset, get_sent_embeddings


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)
BATCHSIZE = 1024 if device=='cuda' else 32
NEPOCHS = 20 if device=='cuda' else 10

def prepare_datasets(twitter_data, tokenizer1, encoder1, tokenizer2=None, encoder2=None):
    twitter_data = twitter_data.dropna(subset='tweets')
    indices = np.arange(len(twitter_data))
    np.random.shuffle(indices)
    train_ind, val_ind, test_ind = np.split(indices, [int(len(indices)*0.7), int(len(indices)*0.85)]) # 70-15-15 train/val/test split
    
    split_inds = {'train': train_ind, 'val': val_ind, 'test': test_ind}
    data_splits = {'train': {}, 'val': {}, 'test': {}}

    sentences = twitter_data['tweets'].values
    sentiment_scores = twitter_data['sentiment'].values

    t0 = time.time()
    print('Encoding sentences...')
    for split in split_inds:
        inds = split_inds[split]
        data_splits[split]['embeddings'] = get_sent_embeddings(tokenizer1, encoder1, tokenizer2, encoder2, device, sentences[inds].tolist())
        data_splits[split]['sent_labels'] = sentiment_scores[inds]
    print(f'Done in {round(time.time() - t0, 2)} seconds')
        
    return TweetDataset(data_splits['train']), TweetDataset(data_splits['val']), TweetDataset(data_splits['test'])


def init_model(hparams, n_classes=3):
    input_dim = hparams['input_dim']
    hidden_dim = hparams['hidden_dim']
    learning_rate = hparams['learning_rate']
    
    model = NeuralSentimentClassifier(input_dim, hidden_dim, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer 


def train(model, optimizer, criterion, train_dataset, val_dataset, hparams):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    model = model.to(device)

    for epoch in range(hparams['num_epochs']):
        train_losses_sub = []
        
        for embeddings_train, labels_train in train_loader:
            optimizer.zero_grad()
            train_outputs = model.forward(embeddings_train.to(device))
            train_loss = criterion(train_outputs, labels_train.to(device))
            
            train_loss.backward()
            optimizer.step()
            
            train_losses_sub.append(train_loss.item())

        epoch_loss = np.mean(train_losses_sub)
        epoch_train_losses.append(epoch_loss)

        epoch_val_acc, epoch_val_loss = eval(model, criterion, val_dataset, hparams)
        epoch_val_accs.append(epoch_val_acc)
        epoch_val_losses.append(epoch_val_loss)
    
    return epoch_train_losses, epoch_val_losses, epoch_val_accs


def accuracy(preds, trues):
    matches = preds.to(device) == trues.to(device)
    return matches.sum().item() / len(trues)


def eval(model, criterion, val_dataset, hparams):
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)
    val_accs = []
    val_losses = []
    
    model.eval()
    with torch.no_grad():
        for embeddings_val, labels_val in val_loader:
            val_outputs = model.forward(embeddings_val.to(device))
            val_loss = criterion(val_outputs, labels_val.to(device))

            val_preds = torch.argmax(val_outputs, dim=1)
            val_accs.append(accuracy(val_preds, labels_val))
            val_losses.append(val_loss.item())
    
    return np.mean(val_accs), np.mean(val_losses)


def train_classifier(train_dataset, val_dataset, hparams):

    model, criterion, optimizer = init_model(hparams)
    epoch_train_losses, epoch_val_losses, epoch_val_accs = train(model, optimizer, criterion, train_dataset, val_dataset, hparams)

    epoch_results = pd.DataFrame()
    epoch_results['epoch'] = np.arange(len(epoch_train_losses))
    epoch_results['train_loss'] = epoch_train_losses
    epoch_results['val_loss'] = epoch_val_losses
    epoch_results['val_acc'] = epoch_val_accs
    for hparam in hparams:
        epoch_results[hparam] = hparams[hparam]
    epoch_results['timestamp'] = int(time.time())
    
    if not os.path.exists('experiment_results.csv'):
        epoch_results.to_csv('experiment_results.csv', index=False)
    else:
        epoch_results.to_csv('experiment_results.csv', mode='a', index=False, header=False)


def get_hparams(args, input_dim):
    
    if args.tune == True:
        hidden_dims = [64, 128, 256, 512]
        learning_rate = [5e-6, 5e-5, 5e-4, 5e-3, 5e-2]
        num_epochs = [NEPOCHS]
        batch_size = [BATCHSIZE]
        n_layers = [0, 1, 2, 3, 4]

        hparam_sets = []
        for hparams in list(itertools.product(*[hidden_dims, learning_rate, num_epochs, batch_size, n_layers])):
            hparam_sets.append({'hidden_dim': hparams[0], 
                                'learning_rate': hparams[1], 
                                'num_epochs': hparams[2],
                                'batch_size': hparams[3],
                                'n_layers': hparams[4],
                                'model': args.model,
                                'input_dim': input_dim})
    else:
        hparam_sets = [{'hidden_dim': 256, 'learning_rate': 5e-3, 'num_epochs': 20, 'batch_size': 32, 'n_layers': 0, 'model': args.model, 'input_dim': input_dim}]

    return hparam_sets


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", choices=['simcse', 'bert', 'both'])
    arg_parser.add_argument("--path_to_data", type=str, default="tw_sentiment_df.csv")
    arg_parser.add_argument("--tune", action="store_true")
    
    return arg_parser.parse_args()


if __name__ == '__main__':
    '''
    Example usage: python train_sentiment_classifier.py simcse
    '''
    
    args = parse_args()
    models = {'simcse': 'princeton-nlp/sup-simcse-bert-base-uncased',
              'bert': 'bert-base-uncased'}
    twitter_data = pd.read_csv(args.path_to_data)
    

    if args.model == 'simcse' or args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(models[args.model])
        encoder = AutoModel.from_pretrained(models[args.model]).to(device)
        train_dataset, val_dataset, test_dataset = prepare_datasets(twitter_data, tokenizer, encoder)
        input_dim = encoder.embeddings.token_type_embeddings.embedding_dim
        hparam_sets = get_hparams(args, input_dim)
    
    elif args.model == 'both':
        tokenizer1 = AutoTokenizer.from_pretrained(models['simcse'])
        encoder1 = AutoModel.from_pretrained(models['simcse']).to(device)
        tokenizer2 = AutoTokenizer.from_pretrained(models['bert'])
        encoder2 = AutoModel.from_pretrained(models['bert']).to(device)
        train_dataset, val_dataset, test_dataset = prepare_datasets(twitter_data, tokenizer1, encoder1, tokenizer2, encoder2)
        input_dim = encoder1.embeddings.token_type_embeddings.embedding_dim + encoder2.embeddings.token_type_embeddings.embedding_dim
        hparam_sets = get_hparams(args, input_dim)
    
    else:
        raise ValueError(f'Invalid choice of model: {args.model}')

    for hparams in tqdm(hparam_sets, desc=f'running tuning experiments'):
        train_classifier(train_dataset, val_dataset, hparams)
    
    
    
            
