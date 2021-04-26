from tweepy import OAuthHandler
import csv
import json
import logging
import os
import time
import tweepy

consumer_key = 'MzGbuu15wRkKPopirC6QUiVPd'
consumer_secret = '2ImFGkcJNXBU0c9bH682U6D3UyOcp3x30vGEqM3Ie782MUdIBZ'
access_token = '1311423992125816832-fGNSZ1lsLJSkgNxfZITAcqLHETM39F'
access_secret = 'ihy1XVrx90Es0abJq7UCPhiGV2OPpRKcA9dmPumLZtdEJ'
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


def getFollowers(name):
    # user_ids = []
    # for page in tweepy.Cursor(api.followers_ids,
    #                           screen_name=name).pages():
    #     user_ids.extend(page)
    # return user_ids
    try:
        user_ids = api.followers_ids(name)
        return user_ids
    except tweepy.TweepError as e:
        return "invalid"


def main():
    NODES = {}  # dict
    nodefile = open('users111.csv')
    nodeline = nodefile.readline()
    while nodeline:
        nodeline = nodeline.strip('\n')
        nodeline = nodeline.split(',')
        NODES[nodeline[0]] = nodeline[1]
        nodeline = nodefile.readline()
    nodefile.close()

    with open("tempfollowers.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["user_id", "followers"])
        for x in NODES.keys():
            user_ids = getFollowers(NODES[x])
            if user_ids != 'invalid':
                write.writerow([x, user_ids])
            print(user_ids)


if __name__ == "__main__":
    main()
