import numpy as np
from tqdm import tqdm_notebook as tqdm
from datetime import datetime

import tweepy

import pandas as pd
from pandas import DataFrame

from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict,cross_validate

def scrap_all_tweets(hashtag,since, until, save_dir, consumer_key, consumer_secret, access_token, access_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, retry_count=5, retry_delay=100000, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    base_tweets = []
    new_tweets = api.search(q=hashtag, since=since, until=until, count=100)
    base_tweets.extend(new_tweets)
    oldest = base_tweets[-1].id - 1
    print("Starting to collect tweets")
    while (len(base_tweets) < 20000) and (len(new_tweets) > 0):
        print( "scraped tweets before %s" % (oldest))
        new_tweets = api.search(q=hashtag, since=since, until=until, count=100, max_id=oldest)
        base_tweets.extend(new_tweets)
        oldest = base_tweets[-1].id - 1
        print( "...%s tweets scraped so far" % (len(base_tweets)))
    
    print("Processing tweets")
    f0 = np.array([tweet.id for tweet in base_tweets])
    f1 = np.array([int(datetime.timestamp(tweet.created_at)) for tweet in base_tweets])
    f2 = np.array([tweet.created_at for tweet in base_tweets])
    # f3 = np.array([tweet.text for tweet in base_tweets])
    f4 = np.array([tweet.user.screen_name for tweet in base_tweets])
    f5 = np.array([tweet.user.followers_count for tweet in base_tweets])
    f6 = np.array([tweet.lang for tweet in base_tweets])
    f7 = np.array([tweet.user.location for tweet in base_tweets])
    f8 = np.array([tweet.place.country if tweet.place != None else None for tweet in base_tweets])
    f9 = np.array([tweet.retweet_count for tweet in base_tweets])
    f10 = np.array([tweet.favorite_count for tweet in base_tweets])
    f11 = np.array([tweet.text.startswith("RT") for tweet in base_tweets])
    f12 = np.array([len(tweet.entities['user_mentions']) for tweet in base_tweets])
    f13 = np.array([len(tweet.entities['hashtags']) for tweet in base_tweets])
    f14 = np.array([len(tweet.entities['urls']) for tweet in base_tweets])
    # f15 = np.array([tweet.author.screen_name for tweet in base_tweets])

    print("Making the dataset")
    start_time = min(f1)
    hrs_passed = int((max(f1)-min(f1))/60)+1
    hr_no_of_tweets = [0] * hrs_passed
    hr_no_of_tweets = [0] * hrs_passed
    hr_no_of_retweets = [0] * hrs_passed
    hr_sum_of_followers = [0] * hrs_passed
    hr_max_no_of_followers = [0] * hrs_passed
    hr_no_of_url_citations = [0] * hrs_passed
    hr_no_of_users = [0] * hrs_passed
    hr_user_set = [0] * hrs_passed
    hr_no_of_mentions = [0] * hrs_passed
    hr_no_of_hashtags = [0] * hrs_passed
    for i in range(0, hrs_passed):
      hr_user_set[i] = set([])
    start_time = min(f1)
    num_data = len(base_tweets)
    for i in tqdm(range(0, num_data)):
        current_hr = int((f1[i]-start_time)/60)
        hr_no_of_tweets[current_hr] += 1
        if f11[i]:
            hr_no_of_retweets[current_hr] += 1
        if f5[i] > hr_max_no_of_followers[current_hr]:
            hr_max_no_of_followers[current_hr] = f5[i]
        hr_sum_of_followers[current_hr] += f5[i]
        hr_no_of_url_citations[current_hr] += f14[i]
        hr_no_of_mentions[current_hr] += f12[i]
        hr_no_of_hashtags[current_hr] += f13[i]
        hr_user_set[current_hr].add(f4[i])

    for i in range(0, len(hr_user_set)):
        hr_no_of_users[i] = len(hr_user_set[i])

    target = hr_no_of_tweets[1:]
    target.append(0)

    data = np.array([hr_no_of_tweets,
                     hr_no_of_retweets,
                     hr_sum_of_followers,
                     hr_max_no_of_followers,
                     hr_no_of_url_citations,
                     hr_no_of_users,
                     hr_no_of_mentions,
                     hr_no_of_hashtags,
                     target])
    data = np.transpose(data)
    
    data_frame = DataFrame(data)
    data_frame.columns = ['no_of_tweets', 
                          'no_of_retweets', 
                          'sum_of_followers',
                          'max_no_of_followers',
                          'no_of_URLs',
                          'no_of_users',
                          'no_of_mentions',
                          'no_of_hashtags',
                          'target']
    
    print("Saving the dataset")
    if os.path.isdir('./Extracted_data'):
        pass
    else:
        os.mkdir('./Extracted_data')
    data_path = os.path.join(save_dir,hashtag+'.csv')
    data_frame.to_csv(data_path, index = False)

def perform_regression(X_train, Y_train):
    reg_rf = RandomForestRegressor(n_estimators = 50, max_depth = 9)
    output1 = cross_validate(reg_rf,X_train,Y_train,return_estimator=True,return_train_score=True)
    cross_validation_values = output1['test_score']
    print("Hashtag --> ",hashtag)
    print('Random Forest --> Min cross-validation error: ',np.abs(np.min(output1['test_score'])))

    reg_lr = LinearRegression(fit_intercept = False) 
    output2 = cross_validate(reg_lr,X_train,Y_train,return_estimator=True,return_train_score=True)
    cross_validation_values = output2['test_score']
    print('Linear Regression --> Min cross-validation error: ',np.abs(np.min(output2['test_score'])))
    
    reg_svm = SVR(kernel='rbf')
    output3 = cross_validate(reg_svm,X_train,Y_train,return_estimator=True,return_train_score=True)
    cross_validation_values = output3['test_score']
    print('Non-linear SVM --> Min cross-validation error: ',np.abs(np.min(output3['test_score'])))

    idx = np.argmin(np.abs(output1['test_score']))
    clf1 = output1['estimator'][idx]

    idx = np.argmin(np.abs(output2['test_score']))
    clf2 = output2['estimator'][idx]

    idx = np.argmin(np.abs(output3['test_score']))
    clf3 = output3['estimator'][idx]

    return [clf1,clf2,clf3]