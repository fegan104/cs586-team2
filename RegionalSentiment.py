import re
from textblob import TextBlob
import json


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        return analysis.sentiment.polarity

    def get_tweets(self):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        # call twitter api to fetch tweets
        tweet_data = '[{"created_at": "Tue Nov 29 18:46:05 +0000 2016","text": "some text that I tweeted", "retweet_count": 0, "user": {"location": "New York, NY"}},{"created_at": "Tue Nov 29 18:46:05 +0000 2016","text": "I am so happy!", "retweet_count": 0, "user": {"location": "New York, NY"}}]'

        fetched_tweets = json.loads(tweet_data)
        # parsing tweets one by one
        for tweet in fetched_tweets:
            # empty dictionary to store required params of a tweet
            parsed_tweet = {}

            # saving text of tweet
            parsed_tweet['time'] = tweet["created_at"]
            parsed_tweet['location'] = tweet["user"]["location"]
            # saving sentiment of tweet
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


def main():
    client = TwitterClient()

    for t in client.get_tweets():
        print t


if __name__ == "__main__":
        # calling main function
    main()
