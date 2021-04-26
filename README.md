# cs298-project
Cyberbullying Classification based on social network relationship features and text features.
Thanks https://github.com/ZeerakW/hatespeech for providing raw dataset. our dataset is expanded based on the hatespeech dataset
dataset: 
1. tweet_dataset_edited.csv contains 'tweet_id,tweet_content,user_id,user_name,category,retweet_user_id_list'
2. followerInfo.csv contians 'user_id, followers', which is users and their followers. it is used to analyze social network relationship features
3. userInfoWithLabelAndRetweetList.csv contains 'user_id, lable_of_posted_tweet, retweet_list_of_posted_tweet'. for each user, we collected all label of their posted tweets and retweet list of posted tweets. We use these information to analyze the user relationships and identify whether a user is bullying user or not.
4. userInfoWithUserName.csv provides user_id and its name.
The way to collect dataset is mainly introduced in twitter_dataset_collect.py and access_followers.py. 
