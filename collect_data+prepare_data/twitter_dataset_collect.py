from tweepy import OAuthHandler
import csv
import json
import logging
import os
import time
import tweepy

logger = logging.getLogger('TwitterAccess')
SLEEP_TIME = 1000


class TwitterDataSet:
    consumer_key = 'T35ai1b9pf9vxeUcYhhqEuAU3'
    consumer_secret = 'IDQdQWs3TGgeqElIQYXYPOiICyUmYj5wgg5qf15Hstp2rrO1Kb'
    access_token = '1311423992125816832-F7a4EbYykGTTzvsvzquVvqZEGdogfZ'
    access_secret = 'Hyh8w6To0et3XlWlkodlUaDTQjdGJpDsSN40ljatTCiyW'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # get twitter context
    def tweet_text_from_tweet_id(self, idx):
        try:
            tweet = self.api.get_status(idx)
            return tweet.text
        except tweepy.TweepError as e:
            return []

    # get follower's user id
    def get_followers(self, screen_name):
        user_ids = []
        for page in tweepy.Cursor(self.api.followers_ids,
                                  screen_name=screen_name).pages():
            user_ids.extend(page)
            time.sleep(30)
        return user_ids

    # get user id from tweeter id
    def user_id_from_tweet_id(self, idx):
        try:
            status = self.api.get_status(idx)
            return status.user.id_str
        except tweepy.TweepError as e:
            return "invalid"

    # get user name from tweet id
    def user_name_from_tweet_id(self, idx):
        status = self.api.get_status(idx)
        return status.user.screen_name

    # check whether two users are friends or not
    def get_follow_info(self, x, y):
        return self.api.show_friendship(source_id=x, target_id=y)

    # get user name from user id
    def user_name_from_user_id(self, idx):
        user = self.api.get_user(user_id=idx)
        return user.screen_name

    def timeline_from_username(self, screen_name):
        timeline = self.api.user_timeline(screen_name=screen_name)
        return timeline

    # get retweet user id from tweet id
    def get_retweet_user_id(self, idx):
        retweet_id_list = self.api.retweets(idx)
        retweet_user_ids = []
        for retweet in retweet_id_list:
            id = retweet.user.id_str
            retweet_user_ids.append(id)
        return retweet_user_ids

    def get_retweet_user_name(self, idx):
        # flag = self.api.get_status
        # print(flag)
        retweet_name_list = self.api.retweets(idx)
        retweet_user_names = []
        for retweet in retweet_name_list:
            name = retweet.user.screen_name
            retweet_user_names.append(name)
        return retweet_user_names

    # 0 (racism), 1 (sexism), or 2 (none)
    def load_raw_data(self):
        tweet_ids = []
        annotations = []
        with open("twitter_data_raw.csv", encoding='utf-8') as csvreadfile:
            read = csv.reader(csvreadfile)
            header = next(read)
            for line in read:
                # print(line)
                # id, annotation = line.split(',')
                tweet_ids.append(line[0])
                annotations.append(line[1])
        return tweet_ids, annotations


# twitter_data_raw.csv: (tweetid, annotation) total 16,202 items
def main():
    twitter_dataset = TwitterDataSet()
    (tweetID, annotations) = twitter_dataset.load_raw_data()
    with open("tweet_info.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["tweet_id", "tweet_content", "user_id", "user_name", "category", "retweet_user_id_list"])
        for i in range(len(tweetID)):
            id = tweetID[i]
            userId = twitter_dataset.user_id_from_tweet_id(id)
            if userId != 'invalid':
                content = twitter_dataset.tweet_text_from_tweet_id(id)
                userName = twitter_dataset.user_name_from_tweet_id(id)
                category = annotations[i]
                retweet_user_id_list = twitter_dataset.get_retweet_user_id(id)
                write.writerow([id, content, userId, userName, category, retweet_user_id_list])
    # retweet_user_id_list = twitter_dataset.get_retweet_user_id(563085860078628865)
    # print(retweet_user_id_list)


if __name__ == "__main__":
    main()
