#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:01:37 2022

@author: jinishizuka
"""

import re 
import tweepy 
from tweepy import OAuthHandler 
from textblob import TextBlob 
import pandas as pd
from better_profanity import profanity


#Twitter developer keys and tokens
consumer_key = 'JJx1idjQNber5YWEiyABUc1zB'
consumer_secret = 'Q0tCzowpFHjzqzRmj5vJ75bTSmkcdWr9tazq9fcKdsPdiT3a5i'
access_token = '928358549054545921-54yjeXEHIWskQciTtcm8dM9BUszGCz7'
access_token_secret = 'h0J44RtSKQOkHa04jW8mkNpq0JxL8F1lyMlNkS2yVZZvt'


#Access twitter data
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Define query
query = 'Elon Musk'

#Define number of tweets to pull
num_tweets = 10


#Filter query to remove retweets
filtered = query + '-filter:retweets'

#Generate the latest tweets on the given query
tweets = tweepy.Cursor(api.search_tweets, 
                           q=filtered,
                           lang="en").items(num_tweets)

# Create a list of the tweets, the users, and their location
list1 = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]


# Convert the list into a dataframe
df = pd.DataFrame(data=list1, 
                    columns=['tweets','user', "location"])


# Convert only the tweets into a list
tweet_list = df.tweets.to_list()


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


cleaned = [clean_tweet(tw) for tw in tweet_list]

# Define the sentiment objects using TextBlob
sentiment_objects = [TextBlob(tweet) for tweet in cleaned]

# Create a list of polarity values and tweet text
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

# Create a dataframe of each tweet against its polarity
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])

# Save the polarity column as 'n'.
n = sentiment_df["polarity"]

# Convert this column into a series, 'm'. 
m = pd.Series(n)

# Initialize variables, 'pos', 'neg', 'neu'.
pos = 0
neg = 0
neu = 0

# Create a loop to classify the tweets as Positive, Negative, or Neutral.
# Count the number of each.

sentiment = []

for items in m:
    if items>0:
        sentiment.append(1)
        pos=pos+1
    elif items<0:
        sentiment.append(-1)
        neg=neg+1
    else:
        sentiment.append(0)
        neu=neu+1

sentiment_df['sentiment'] = sentiment
        
print('Num positive: {}, Num negative: {}, Num nuetral: {}'.format(pos,neg,neu))


#save to csv
sentiment_df.to_csv('tw_sentiment_df.csv')








