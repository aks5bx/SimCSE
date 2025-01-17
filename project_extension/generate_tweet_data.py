#############
## IMPORTS ##
#############

import re 
from argparse import ArgumentParser
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import pandas as pd
from better_profanity import profanity
from tqdm import tqdm
import searchtweets
from searchtweets import ResultStream, gen_request_parameters, load_credentials, collect_results
from tweet_parser.tweet import Tweet
import os
import yaml 
import json 

##################
## CONFIG SETUP ##
##################

f = open('project_extension/tw_premium_api.json')
premium_dict = json.load(f)['Premium Info']

#Twitter developer keys and tokens
consumer_key =  premium_dict['consumer_key'] #'JJx1idjQNber5YWEiyABUc1zB'
consumer_secret = premium_dict['consumer_secret']  # 'Q0tCzowpFHjzqzRmj5vJ75bTSmkcdWr9tazq9fcKdsPdiT3a5i'
access_token = premium_dict['access_token']  # '928358549054545921-54yjeXEHIWskQciTtcm8dM9BUszGCz7'
access_token_secret = premium_dict['access_token_secret']  #'h0J44RtSKQOkHa04jW8mkNpq0JxL8F1lyMlNkS2yVZZvt'

#######################
## TWITTER API SETUP ##
#######################

#Access twitter data
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, retry_count=10, retry_delay=5)

def premium_setup(yaml_path):
    search_args = load_credentials(yaml_path,
                                   yaml_key="search_tweets_v2",
                                   env_overwrite=False)
    
    return search_args    

# Create a function to clean the tweets. Remove profanity, unnecessary characters, spaces, and stopwords.
def clean_tweet(tweet):
    if type(tweet) == float:
        return ""
    r = tweet.lower()
    r = profanity.censor(r)
    r = re.sub("'", "", r) # This is to avoid removing contractions in english
    r = re.sub("@[A-Za-z0-9_]+","", r)
    r = re.sub("#[A-Za-z0-9_]+","", r)
    r = re.sub(r'http\S+', '', r)
    r = re.sub('[()!?]', ' ', r)
    r = re.sub('\[.*?\]',' ', r)
    r = re.sub("[^a-z0-9]"," ", r)
    r = r.split()
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    r = [w for w in r if not w in stopwords]
    r = " ".join(word for word in r)
    return r

def query_data_premium_v2(query, search_args, num_tweets, stream=True):
    query = gen_request_parameters(query, 
                                   granularity = None, 
                                   start_time = '2022-11-26',
                                   end_time = '2022-12-03',
                                   results_per_call=50)

    if stream:
        print('USING METHOD : Result Stream')
        rs = ResultStream(request_parameters=query,
                          max_tweets=num_tweets,
                          max_requests=None,
                          **search_args)

        tweets = list(rs.stream())
        print('Tweets Len:', len(tweets))
    else:
        print('USING METHOD : Collect Results')
        tweets = collect_results(query,
                                max_tweets=100,
                                result_stream_args=search_args)

    # Create a list of the tweets, the users, and their location
    results = []
    for tweet in tqdm(tweets):
        for tweet_data in tqdm(tweet['data']):
            tweet_info = [tweet_data['text'], tweet_data['id'], '0']
            results.append(tweet_info)

    # Convert the list into a dataframe
    df = pd.DataFrame(data=results, 
                        columns=['tweets','user', "location"])

    print('')
    print(len(results))

    # Convert only the tweets into a list
    tqdm.pandas(desc='cleaning data')
    df['tweets'] = df['tweets'].progress_apply(lambda x: clean_tweet(x))

    # Convert the list into a dataframe
    df = pd.DataFrame(data=results, 
                        columns=['tweets','user', "location"])

    # Convert only the tweets into a list
    tqdm.pandas(desc='cleaning data')
    df['tweets'] = df['tweets'].progress_apply(lambda x: clean_tweet(x))

    return df, df['tweets'].values.tolist()    

def query_data(query, num_tweets):

    #Filter query to remove retweets
    filtered = query + '-filter:retweets'

    #Generate the latest tweets on the given query
    print('querying twitter data...')
    tweets = tweepy.Cursor(api.search_tweets, 
                            q=filtered,
                            lang="en").items(num_tweets)

    # Create a list of the tweets, the users, and their location
    results = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]

    # Convert the list into a dataframe
    df = pd.DataFrame(data=results, 
                        columns=['tweets','user', "location"])

    # Convert only the tweets into a list
    tqdm.pandas(desc='cleaning data')
    df['tweets'] = df['tweets'].progress_apply(lambda x: clean_tweet(x))

    return df, df['tweets'].values.tolist()

def bin_polarity(score):
    if score > 0:
        sentiment = 2
    elif score < 0:
        sentiment = 0
    else:
        sentiment = 1
    return sentiment

def get_sentiment_scores(query, num_tweets, premium=True):
    if premium:
        search_args = premium_setup('project_extension/twitter_keys.yaml')
        tweets_df, tweets = query_data_premium_v2(query, search_args, num_tweets, stream=True)
        print('Sent Score :', len(tweets))
    else:
        tweets_df, tweets = query_data(query, num_tweets)
    
    # Define the sentiment objects using TextBlob
    sentiment_objects = [TextBlob(tweet) for tweet in tweets]
    
    # Get polarity values and bin them into categories (positive (1), neutral (0), negative (-1))
    polarities = [tweet.sentiment.polarity for tweet in sentiment_objects]
    tweets_df['polarity'] = polarities

    tqdm.pandas(desc='getting sentiment labels')
    tweets_df['sentiment'] = tweets_df['polarity'].progress_apply(lambda x: bin_polarity(x))

    label_dict = {'pos': 2, 'neg': 0, 'neu': 1}
    for label in label_dict:
        pass
        #print(f'{label} count:', len(tweets_df[tweets_df['sentiment']==label_dict[label]]))

    #save to csv
    tweets_df.to_csv('tw_sentiment_df_' + str(num_tweets) + '.csv', index=False)

def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("query", type=str)
    arg_parser.add_argument("country_subset", type=str)
    arg_parser.add_argument("num_tweets", type=int)

    return arg_parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.country_subset == 'subset':
        query = 'fifa' + ' ('
        for country in ['england', 'iran', 'usa', 'wales', 
                        'france', 'australia', 'denmark', 'tunisia', 
                        'brazil', 'serbia', 'switzerland', 'cameroon']:
            query = query + country + ' OR ' + country.capitalize() + ' OR '
        
        query = query + 'FIFA) lang:en'

        print('Using Query:')
        print(query)
        print('--' * 75)
        get_sentiment_scores(query, args.num_tweets, premium=True)
    else:
        get_sentiment_scores(args.query, args.num_tweets, premium=True)


