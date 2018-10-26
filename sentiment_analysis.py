import tweepy
from textblob import TextBlob

consumer_key = '5KnefA5n4pfa1Yy57XM8cjmlj'
consumer_secret = 'GawGO3li2pza53k9ArlBriGPeZOabhL3dnNGFK6sKK14hIQRlG'
access_token = '3067325342-58BsTGjv1sbt9u50TEfKXPFonwNpNJoctZsGzof'
access_token_secret = 'DeRZ34D1Twivqhx8M4tL3lWsTw839Y2PLFjsubPFphUc8'
# OWNER ID = 3067325342

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Bitcoin')
sentiment_value = 0
subjectivity_value = 0
for tweet in public_tweets:
    #print(tweet.text)
    analysis = TextBlob(tweet.text)
    final_sentiment = (analysis.sentiment[0])
    final_subjectivity = analysis.sentiment[1]
    #print(analysis.sentiment)
    #print(final_sentiment)
    sentiment_value += final_sentiment
    subjectivity_value += final_subjectivity
    #print(sentiment_value)
final_sentiment_value = (sentiment_value / 2)
final_subjectivity_value = (subjectivity_value / 2) # This number may need to be the number of call made
print(final_sentiment_value)
print(final_subjectivity_value)
