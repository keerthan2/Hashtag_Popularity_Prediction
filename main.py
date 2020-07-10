import os
import matplotlib.pyplot as plt
from utils import *

consumer_key = # Add your API key here
consumer_secret =  # Add your API secret key here
access_token = # Add your access token
access_secret = # Add your access secret

hashtag = "#DarkNetflix"
download_data = True
since = "2020-06-27" # ignore if download_data is False. Must be utmost 20 days from the current date
until = "2020-06-28" # ignore if download_data is False. Must be utmost 20 days from the current date
save_dir = './data' # Directory to save the data. 

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if download_data:
    scrap_all_tweets(hashtag,since, until, save_dir, consumer_key, consumer_secret, access_token, access_secret)

data_path = os.path.join(save_dir,hashtag+'.csv')
train_split = 0.8
training_data = pd.read_csv(data_path)
target_data = training_data.pop('target')
n_train = int(train_split*len(training_data))

X_train = training_data[:n_train]
Y_train = target_data[:n_train]
X_test = training_data[n_train:]
Y_test = target_data[n_train:]

clfs = perform_regression(X_train,Y_train)

clf_id = 1 # 0:RF, 1:LR, 2:SVM
Y_pred = clfs[clf_id].predict(X_test)
plt.plot(range(len(Y_pred)),Y_pred,range(len(Y_pred)),Y_test)
plt.legend(['Predicted number of tweets','Actual number of tweets'])
plt.title(hashtag+" predictions")
plt.ylabel("Number of tweets")
plt.xlabel("Minutes")
plt.show()
