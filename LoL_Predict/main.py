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
my_region = 'eun1'

me = watcher.summoner.by_name(my_region, 'Positivity')
print(me)
pId={}
#Mecevi

my_matches = watcher.match.matchlist_by_account(my_region, me['accountId'])
    

participants = []

for i in range(100):
    last_match = my_matches['matches'][i]
    match_detail = watcher.match.by_id(my_region, last_match['gameId'])   
    for row in match_detail:
        for r in match_detail['participantIdentities']:
            if r['player']['accountId'] == me['accountId']:
                new_value = r['participantId']
    #Statistike za gejm ciji je gameId pozvan
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
            participants.append(participants_row)


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
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_predictions = lr.predict(X_test)
print(classification_report(y_test,lr_predictions))
print(confusion_matrix(y_test,lr_predictions))


