import pandas as pd
import pickle, os, json
import parse from dateutil.parser
from os import walk

def get_tweets(self, dirname, filename):
        '''
        Main function to fetch tweets and parse them.
        '''
        # call twitter api to fetch tweets
        fetched_tweets = []
        with open(dirname+"/"+filename, 'r') as f:
                for line in f:
                fetched_tweets.append(json.loads(line))

        # return parsed tweets
        return fetched_tweets

def main():
        cities = ['Atlanta','Boston']
        months = {3: 0, 9: 1, 11:2}
        month_lengths = {3: 30, 9: 31, 11: 31}
        for city in cities:
                df = pickle.load(open("weather/combined/{}.pkl".format(city),"rb"))
                print(list(df))

                df['date'] = df['date'].apply(parse)

                # Get a dictionary of all directory mapped to their files
                files = {}
                for (dirpath, _, filenames) in walk("./clean-location-data"):
                        if dirpath in files:
                                files[dirpath].append(filenames)
                        elif len(filenames) > 0:
                                files[dirpath] = filenames

                for d in files:
                        for f in files[d]:
                                for t in get_tweets(d, f):
                                        if t['location'][:4] == city[:4]:
                                                tdt = parse(t['time'])
                                                date = tdt.date()
                                                index = months[date.month]*month_lengths[date.month]*24 + date.day*24 + date.hour
                                                print(df.iloc[index])

if __name__ == "__main__":
        # calling main function
        main()