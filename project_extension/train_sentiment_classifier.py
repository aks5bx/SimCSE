from argparse import ArgumentParser
from functools import partial
import itertools
import os
import time
import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from neural_sentiment_classifier import NeuralSentimentClassifier
from tweet_dataset import TweetDataset, tweet_batch_collate


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)
BATCHSIZE = 1024 if device=='cuda' else 32
NEPOCHS = 25 if device=='cuda' else 10

def prepare_datasets(twitter_data, n_samples):
    twitter_data = twitter_data.dropna(subset='tweets')
    indices = np.arange(len(twitter_data))
    np.random.shuffle(indices)
    train_ind, val_ind, test_ind = np.split(indices, [int(len(indices)*0.7), int(len(indices)*0.85)]) # 70-15-15 train/val/test split
    
    split_inds = {'train': train_ind, 'val': val_ind, 'test': test_ind}
    data_splits = {'train': {}, 'val': {}, 'test': {}}

    sentences = twitter_data['tweets'].values
    sentiment_scores = twitter_data['sentiment'].values

    for split in split_inds:
        inds = split_inds[split]
        data_splits[split]['sentences'] = sentences[inds]
        data_splits[split]['sent_labels'] = sentiment_scores[inds]
        if split == 'train':
            data_splits[split]['sentences'] = data_splits[split]['sentences'][:n_samples]
            data_splits[split]['sent_labels'] = data_splits[split]['sentences'][:n_samples]
        
    return TweetDataset(data_splits['train']), TweetDataset(data_splits['val']), TweetDataset(data_splits['test'])


def init_model(hparams, n_classes=3):
    input_dim = hparams['input_dim']
    hidden_dim = hparams['hidden_dim']
    learning_rate = hparams['learning_rate']
    
    model = NeuralSentimentClassifier(input_dim, hidden_dim, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer 


def train(model, optimizer, criterion, train_dataset, val_dataset, batch_collater, hparams, batch_collater_val=None):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accs = []

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], collate_fn=batch_collater, shuffle=True)
    for epoch in range(hparams['num_epochs']):
        train_losses_sub = []
        
        for embeddings_train, labels_train, sentences_train in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            optimizer.zero_grad()
            train_outputs = model.forward(embeddings_train.to(device))
            train_loss = criterion(train_outputs, labels_train.to(device))
            
            train_loss.backward()
            optimizer.step()
            
            train_losses_sub.append(train_loss.item())

        epoch_loss = np.mean(train_losses_sub)
        epoch_train_losses.append(epoch_loss)

        if batch_collater_val == None:
            epoch_val_acc, epoch_val_loss = eval(model, criterion, val_dataset, hparams, batch_collater)
        else:
            epoch_val_acc, epoch_val_loss = eval(model, criterion, val_dataset, hparams, batch_collater_val)

        epoch_val_accs.append(epoch_val_acc)
        epoch_val_losses.append(epoch_val_loss)
    
    return epoch_train_losses, epoch_val_losses, epoch_val_accs


def accuracy(preds, trues):
    matches = preds.to(device) == trues.to(device)
    return matches.sum().item() / len(trues)


def eval(model, criterion, val_dataset, hparams, batch_collater):
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], collate_fn=batch_collater, shuffle=False)
    val_accs = []
    val_losses = []
    
    model.eval()
    with torch.no_grad():
        for embeddings_val, labels_val, sentences_val in val_loader:
            val_outputs = model.forward(embeddings_val.to(device))
            val_loss = criterion(val_outputs, labels_val.to(device))

            val_preds = torch.argmax(val_outputs, dim=1)
            val_accs.append(accuracy(val_preds, labels_val))
            val_losses.append(val_loss.item())
    
    return np.mean(val_accs), np.mean(val_losses)


def train_classifier(tokenizer1, encoder1, data, hparams, tokenizer2=None, encoder2=None, permute=None, save=False):

    train_dataset, val_dataset, test_dataset = prepare_datasets(data)

    if permute != None:
        batch_collater = partial(tweet_batch_collate, tokenizer1=tokenizer1, encoder1=encoder1, tokenizer2=tokenizer2, encoder2=encoder2, permute1 = permute[0], permute2 = permute[1], device=device)
        batch_collater_val = partial(tweet_batch_collate, tokenizer1=tokenizer1, encoder1=encoder1, tokenizer2=tokenizer2, encoder2=encoder2, permute1 = permute[0], permute2 = permute[1], device=device)
    else:
        batch_collater = partial(tweet_batch_collate, tokenizer1=tokenizer1, encoder1=encoder1, tokenizer2=tokenizer2, encoder2=encoder2, device=device)

    model, criterion, optimizer = init_model(hparams)

    if permute != None:
        epoch_train_losses, epoch_val_losses, epoch_val_accs = train(model, optimizer, criterion, train_dataset, val_dataset, batch_collater, hparams, batch_collater_val)
    else:
        epoch_train_losses, epoch_val_losses, epoch_val_accs = train(model, optimizer, criterion, train_dataset, val_dataset, batch_collater, hparams)

    epoch_results = pd.DataFrame()
    epoch_results['epoch'] = np.arange(len(epoch_train_losses))
    epoch_results['train_loss'] = epoch_train_losses
    epoch_results['val_loss'] = epoch_val_losses
    epoch_results['val_acc'] = epoch_val_accs
    for hparam in hparams:
        epoch_results[hparam] = hparams[hparam]
    epoch_results['timestamp'] = int(time.time())
    epoch_results['model'] = hparams['model']

    if args.exp_output == 'n_sample_experiment_results.csv':
        epoch_results['n_samples'] = args.n_samples
    
    if not os.path.exists(args.exp_output):
        epoch_results.to_csv(args.exp_output, index=False)
    else:
        epoch_results.to_csv(args.exp_output, mode='a', index=False, header=False)

    if args.save == True:
        path = f'{hparams["model"]}_final.pt'
        torch.save(model.state_dict(), path)

    print('Best Validation Accuracy:', np.max(epoch_val_accs))


def get_hparams(args):
    
    if args.tune == True:
        hidden_dims = [128, 256, 512]
        learning_rate = [5e-5, 5e-4, 5e-3]
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
                                'model': args.model})
    else:
        # Set hparams to best parameters found through tuning to date
        best_hparams = json.load(open('best_hparams.json', 'r'))

        if args.model == 'simcse':
            hparams = best_hparams['simcse']
        elif args.model == 'bert':
            hparams = best_hparams['bert']
        elif args.model == 'both':
            hparams = best_hparams['both']
        
        hparams['num_epochs'] = NEPOCHS
        hparams['batch_size'] = BATCHSIZE
        hparam_sets = [hparams]

    return hparam_sets


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", choices=['simcse', 'bert', 'both'])
    arg_parser.add_argument("--path_to_data", type=str, default="tw_sentiment_df_10000.csv")
    arg_parser.add_argument("--tune", action="store_true")
    arg_parser.add_argument("--save", action="store_true")
    arg_parser.add_argument("--permute1", type=str, default="N")
    arg_parser.add_argument("--permute2", type=str, default="N")
    arg_parser.add_argument("--n_samples", type=int, default=10000)
    arg_parser.add_argument("--exp_output", default="n_sample_experiment_results.csv")
    
    return arg_parser.parse_args()


if __name__ == '__main__':
    '''
    Example usage: python train_sentiment_classifier.py simcse
    '''
    
    args = parse_args()
    models = {'simcse': 'princeton-nlp/sup-simcse-bert-base-uncased',
              'bert': 'bert-base-uncased'}

    if args.permute1 == 'N' and args.permute2 == 'N':
        permute = None
    else:
        permute = (args.permute1, args.permute2)

    twitter_data = pd.read_csv(args.path_to_data)

    hparam_sets = get_hparams(args)

    for hparams in tqdm(hparam_sets, desc='experiment'):
        if args.model == 'simcse' or args.model == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(models[args.model])
            encoder = AutoModel.from_pretrained(models[args.model]).to(device)
            hparams['input_dim'] = encoder.embeddings.token_type_embeddings.embedding_dim
            train_classifier(tokenizer, encoder, twitter_data, hparams, args)

        elif args.model == 'both':
            tokenizer1 = AutoTokenizer.from_pretrained(models['simcse'])
            encoder1 = AutoModel.from_pretrained(models['simcse']).to(device)
            tokenizer2 = AutoTokenizer.from_pretrained(models['bert'])
            encoder2 = AutoModel.from_pretrained(models['bert']).to(device)
            hparams['input_dim'] = encoder1.embeddings.token_type_embeddings.embedding_dim + \
                                encoder2.embeddings.token_type_embeddings.embedding_dim
            train_classifier(tokenizer1, encoder1, twitter_data, hparams, tokenizer2, encoder2, permute=permute, save=args.save)
    
        else:
            raise ValueError(f'Invalid choice of model: {args.model}')
    
            
