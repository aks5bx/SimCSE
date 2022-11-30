import pandas as pd 
import os

def explore_data(df):
    print('DF Head')
    print(df.head(5))
    print('--' * 75)

    print('DF INFO')
    print(df.info())

def main():
    sentiment_df = pd.read_csv('tw_sentiment_df.csv')
    explore_data(sentiment_df)

if __name__ == '__main__':
    main()