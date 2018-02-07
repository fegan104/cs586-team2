# pylint: disable=print-statement
import re
from textblob import TextBlob
import json
import os
from os import walk
import errno


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def get_tweet_polarity(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)
        # set sentiment
        return analysis.sentiment.polarity

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


    def get_tweets(self, dirname, filename):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        # call twitter api to fetch tweets
        fetched_tweets = []
        with open(dirname+"/"+filename, 'r') as f:
            for line in f:
                fetched_tweets.append(json.loads(line))

        # parsing tweets one by one
        for tweet in fetched_tweets:
            # empty dictionary to store required params of a tweet
            parsed_tweet = {}

            # saving text of tweet
            parsed_tweet['time'] = tweet["created_at"]
            parsed_tweet['location'] = tweet["user"]["location"]
            # saving sentiment of tweet
            parsed_tweet['polarity'] = self.get_tweet_polarity(tweet["text"])
            parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet["text"])

            # appending parsed tweet to tweets list
            if tweet["retweet_count"] > 0:
                # if tweet has retweets, ensure that it is appended only once
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
            else:
                tweets.append(parsed_tweet)

        # return parsed tweets
        return tweets

def saveTweetsToFile(tweets, dir, file):
    try:
        os.makedirs("./clean-twitter-data/"+dir)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise
    with open("./clean-twitter-data/"+dir+"/"+file, "a+") as fout:
        json.dump(tweets, fout)
        fout.write('\n')

def main():
    client = TwitterClient()

    # Get a dictionary of all directory mapped to their files
    files = {}
    for (dirpath, _, filenames) in walk("./raw-twitter-data"):
        if dirpath in files:
            files[dirpath].append(filenames)
        elif len(filenames) > 0:
            files[dirpath] = filenames

    for d in files:
        print "d is "+ d
        for f in files[d]:
            print "f is "+ f
            for t in client.get_tweets(d, f):
                saveTweetsToFile(t, os.path.basename(d), f)
                print "saving "+d+"/"+f 

if __name__ == "__main__":
    # calling main function
    main()
