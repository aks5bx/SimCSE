import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
import pandas as pd

def get_simcse_embeddings(model, data):
    return None


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", type=str, default="princeton-nlp/sup-simcse-bert-base-uncased")
    arg_parser.add_argument("path_to_data", type=str, default="tw_sentiment_df.csv")

    return arg_parser.parse_args()

def main(model, twitter_data):
    sent_embeddings = get_simcse_embeddings(model, twitter_data)

if __name__ == '__main__':
    args = parse_args()
    simcse_model = SimCSE(args.model)
    twitter_data = pd.read_csv(args.path_to_data)
    main(simcse_model, twitter_data)
