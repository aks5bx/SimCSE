import pandas as pd 
import os

def explore_dataframe(df):
    print('DF Head')
    print(df.head(5))
    print('-' * 75)
    print('DF INFO')
    print(df.info())

def unique_countries(df):
    print('Country Info Breakdown')
    print(df.location.value_counts())
    print('-')
    print('List of Unique Countries')
    print('   ', set(df.location))

def class_imbalance(df):
    print('Class Breakdown')
    print(df.sentiment.value_counts())

def eda_pipeline(df):
    print('-' * 75)
    explore_dataframe(df)
    print('-' * 75)
    unique_countries(df)
    print('-' * 75)
    class_imbalance(df)
    print('-' * 75)

def main():
    sentiment_df = pd.read_csv('tw_sentiment_df.csv')
    eda_pipeline(sentiment_df)

if __name__ == '__main__':
    main()