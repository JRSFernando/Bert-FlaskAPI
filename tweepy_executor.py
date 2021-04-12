# import required libraries
import tweepy
import time
import pandas as pd
pd.set_option('display.max_colwidth', 1000)

# api key
api_key = "5He5g4sPvFzx7ezkBT6yvCdTk"
# api secret key
api_secret_key = "7hOTth3zNMiGswfxVZGimWsclHUIJUhA3oyILV0bQlkHjXLmgY"
# access token
access_token = "1364530423804563456-IVYixRoVDCYTL4NN8zPWsYNYeo25Lb"
# access token secret
access_token_secret = "9U4mnk6HHyrkRibk35dEo4sPLJQESB3DuPQkhq19XXur9"

# authorize the API Key
authentication = tweepy.OAuthHandler(api_key, api_secret_key)

# authorization to user's access token and access token secret
authentication.set_access_token(access_token, access_token_secret)

# call the api
api = tweepy.API(authentication, wait_on_rate_limit=True)

def get_related_tweets(text_query):
    # list to store tweets
    tweets_list = []
    # no of tweets
    count = 5
    try:
        # Pulling individual tweets from query
        for tweet in api.search(q=text_query, count=count):
            # Adding to list that contains all tweets
            tweets_list.append(tweet._json)
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)