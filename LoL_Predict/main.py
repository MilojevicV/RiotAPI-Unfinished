from numpy.lib.function_base import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from riotwatcher import LolWatcher, ApiError

# global variables
api_key = 'RGAPI-cb5bdd0c-a23e-47d8-a918-07ba1ef50ecd'
watcher = LolWatcher(api_key)
my_region=input('Enter the region from which you are playing from (euw1,eun1,na1...):')
summName = input('Enter your summoner name ( Exactly as written ): ')

me = watcher.summoner.by_name(my_region, summName)
print(me)

#Pulling a list of matches

my_matches = watcher.match.matchlist_by_account(my_region, me['accountId'])
    

participants = []

for i in range(100):
    last_match = my_matches['matches'][i]
    match_detail = watcher.match.by_id(my_region, last_match['gameId']) 
    #Identification of the summoner through his participantId 
    for row in match_detail:
        for r in match_detail['participantIdentities']:
            if r['player']['accountId'] == me['accountId']:
                new_value = r['participantId']
    #Pulling certain stats from the match
    for row in match_detail['participants']:
        participants_row = {}
        if row['participantId'] == new_value:
            participants_row['win'] = row['stats']['win']
            participants_row['kills'] = row['stats']['kills']
            participants_row['deaths'] = row['stats']['deaths']
            participants_row['assists'] = row['stats']['assists']
            participants_row['totalDamageDealtToChampions'] = row['stats']['totalDamageDealtToChampions']
            participants_row['totalMinionsKilled'] = row['stats']['totalMinionsKilled']
            participants_row['turretKills'] = row['stats']['turretKills']
            participants_row['damageDealtToObjectives'] = row['stats']['damageDealtToObjectives']
            #Appending stats into a list for future use
            participants.append(participants_row)

#Making the dataframe and spliting data
df = pd.DataFrame(participants)
win = pd.get_dummies(df['win'],drop_first=True)
df.drop(['win'],axis=1,inplace=True)
df = pd.concat([df,win],axis=1)
X = df.drop(True,axis=1)
y= df[True]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Using the model for predictions
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_prediction_data = df.head(10).drop(True,axis=1)
lr_checkup = df.head(10)[True]
lr_predictions = lr.predict(lr_prediction_data)
print(confusion_matrix(lr_checkup,lr_predictions))
print('Predictions were: ', lr_predictions,' while the actual data was: ', lr_checkup)



