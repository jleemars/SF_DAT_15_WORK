# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:08:07 2015

@author: jlmars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier





df = pd.read_csv('../Work/data/USBC_Classi_AllEve-2.csv')


df.head()
df.columns
df.dtypes
df.shape


df['Hometown'].apply(lambda x: x.split(','))
df['City'] = df['Hometown'].apply(lambda x: x.split(',')[0])
df['State'] = df['Hometown'].apply(lambda x: x.split(',')[1])

State = df.groupby('State').count()
State.sort_index(by='Hometown', ascending=False).head(10)


df.plot(kind = 'scatter', x='Singles_Event_Total', y='Doubles_Event_Total')
df.plot(kind = 'scatter', x='Team_Event_Total', y='Doubles_Event_Total')


df.rename(columns={'Team,Doubles,Singles': 'Squad_Totals_TDS'}, inplace=True)


df['Squad_Totals_TDS'].apply(lambda x: x.split(','))
df['Team_Event_Total'] = df['Squad_Totals_TDS'].apply(lambda x: x.split(',')[0])
df['Doubles_Event_Total'] = df['Squad_Totals_TDS'].apply(lambda x: x.split(',')[1])
df['Singles_Event_Total'] = df['Squad_Totals_TDS'].apply(lambda x: x.split(',')[2])

df[['Doubles_Event_Total', 'Team_Event_Total']] = df[['Doubles_Event_Total', 'Team_Event_Total']].astype(float)
df[['Singles_Event_Total']] = df[['Singles_Event_Total']].astype(float)

df['Avg_Total_2015'] = df['Avg_2015']*9
df['Avg_Total_2014'] = df['Avg_2014']*9
df['Avg_Total_2013'] = df['Avg_2013']*9

df['Total_Avg_last3'] = df.Avg_Total_2015+ df.Avg_Total_2014 + df.Avg_Total_2013
df['Avg_last3'] = df['Total_Avg_last3']/3

df['Diff_AvgVSBowled'] = df['Total'] - df['Avg_last3']
df['Avg_last3bygame'] = df['Avg_last3']/9

def avg_range(row):
    data = [row['Avg_2013'], row['Avg_2014'], row['Avg_2015']]
    return max(data) - min(data)
    
df['avg_range'] = df.apply(avg_range, axis=1)
df['avg_range_total'] = df['avg_range']*9
df['Total_avgBG'] = df['Diff_AvgVSBowled']/9


df[['Name', 'Total', 'avg_range']].sort_index(by='avg_range', ascending=False).head(20)


df.plot(kind = 'scatter', x='Avg_last3bygame', y='Diff_AvgVSBowled')

d = np.array(df[['Avg_last3bygame', 'Diff_AvgVSBowled']])
d = np.array(df[['avg_range_total', 'Diff_AvgVSBowled']])
d = np.array(df[['Diff_AvgVSBowled', 'avg_range']])
d = np.array(df[['avg_range', 'Total']])
d = np.array(df[['avg_range', 'Total_avgBG']])


est = KMeans(n_clusters=7, init='random')
est.fit(d)
y_kmeans = est.predict(d)


plt.figure()
plt.scatter(d[:, 0], d[:, 1], c=y_kmeans, s=50)

df['cluster'] = y_kmeans

for cluster in range(1,2):
    print cluster
    print df[df.cluster==cluster]

df[['Name', 'Total','avg_range', 'Total_avgBG']][df.cluster==6]

df[['Name']][df.Log_Bagger_Response==1].count()

'''
cluster2 - 8
cluster6 - 19

'''


df.head()
'''
potetial baggers are clusters x and y
'''

logbaggers = {2:1, 6:1}
df['Log_Bagger_Response'] = df['cluster'].apply(lambda x: logbaggers.get(x,0))

KNNbaggers = {0:0, 1:0, 2:2, 3:0, 4:0, 5:0, 6:1, 7:0}
df['KNN_Bagger_Response'] = df['cluster'].apply(lambda x: KNNbaggers.get(x,0))


df[['Name']][df.KNN_Bagger_Response==0].count()

df.to_csv('Bowling_Proj_Log-knn1.csv', header=True)

'''
Log Reg
'''
Logfeatures = ['Total_avgBG', 'avg_range_total', 'avg_range']
X = df[Logfeatures]
y = df.Log_Bagger_Response
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(Logfeatures, logreg.coef_[0]) 

y_pred = logreg.predict(X_test)

print metrics.accuracy_score(y_test, y_pred)
'''
1.0
'''

scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
'''
array([ 1.        ,  0.98305085,  1.        ,  0.96551724,  0.98275862])
'''

metrics.confusion_matrix(y_test, y_pred)

'''
sensitivity: 5/6 = .83333
specificity: 68/68 = 1
'''

'''
KNN
'''
df.head()
df.columns

KNNfeatures = ['Total_avgBG','Avg_last3bygame','avg_range']
KnnX = df[KNNfeatures]
Knny = df.KNN_Bagger_Response

'''
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(KnnX, Knny)
knn.score(KnnX, Knny)
'''
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, KnnX, Knny, cv=5, scoring='accuracy')
np.mean(scores)

'''
array([ 0.96610169,  0.96610169,  0.93220339,  0.98275862,  0.98275862])

0.96598480420806543

'''











