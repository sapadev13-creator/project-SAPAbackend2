import os

import tweepy
from tweepy import OAuth2UserHandler

from sapa_api.config import (
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_CLIENT_ID,
    TWITTER_REDIRECT_URI,
    TWITTER_SCOPES,
)


def get_twitter_client(access_token, access_token_secret):
    return tweepy.Client(
        consumer_key=TWITTER_API_KEY,
        consumer_secret=TWITTER_API_SECRET,
        access_token=access_token,
        access_token_secret=None,
    )


def get_oauth_handler():
    return OAuth2UserHandler(
        client_id=TWITTER_CLIENT_ID,
        redirect_uri=TWITTER_REDIRECT_URI,
        scope=TWITTER_SCOPES,
    )


oauth = get_oauth_handler()


def fetch_user_tweets(access_token: str, max_results: int = 10):
    client = tweepy.Client(
        client_id=TWITTER_CLIENT_ID,
        client_secret=os.getenv("TWITTER_CLIENT_SECRET"),
        access_token=access_token,
        token_type="user",
    )
    me = client.get_me()
    tweets = client.get_users_tweets(
        id=me.data.id,
        max_results=max_results,
        exclude=["retweets", "replies"],
    )
    if not tweets.data:
        return ""
    return " ".join(t.text for t in tweets.data)
