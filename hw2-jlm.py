##### Part 1 #####


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. read in the yelp dataset

df = pd.read_csv('../hw/optional/yelp.csv')
df.head()
df.tail()
df.describe()
df.index
df.columns
df.shape

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors

feature_cols = ['cool', 'useful', 'funny']
X = df[feature_cols]
y = df.stars


# do train test split on this as well
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

zip(feature_cols, linreg.coef_)

#sns.pairplot(df, x_vars=feature_cols, y_vars='stars', size=6, aspect=0.7, kind='reg')

# 3. Show your MAE, R_Squared and RMSE

y_pred = linreg.predict(X_test)

#R_squared
metrics.r2_score(y_test, y_pred)

#metrics functions

#MAE
metrics.mean_absolute_error(y_test, y_pred)
# .947 - means we may be off by 1 star give or take

#MSE
metrics.mean_squared_error(y_test, y_pred)
#RMSE
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


lm = smf.ols(formula='stars ~ cool + useful + funny', data=df).fit()
lm.rsquared

# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?

print lm.pvalues

'''
cool         2.988197e-90
useful       1.206207e-39
funny        1.850674e-43

As all 3 fall below .05 confidence, we would be ok to keep all 3
'''

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4

df['good_rating']=(df['stars']==4)|(df['stars']==5)
df.columns
df.head()

df[df.good_rating == True].count()
df[df.good_rating == False].count()

# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors


feature_cols = ['cool', 'useful', 'funny']
X = df[feature_cols]
y = df.good_rating
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0]) # what is 0 for?


#sns.pairplot(df, x_vars=feature_cols, y_vars='good_rating', size=6, aspect=0.7, kind='reg')

# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix

y_pred_class = logreg.predict(X_test)

#accuracy
print metrics.accuracy_score(y_test, y_pred_class)

metrics.confusion_matrix(y_test, y_pred_class)
'''
[[  51,  733],
 [  38, 1678]])
 
sensitivity: 1678/1716 = .97785
specificity: 51/784 = .06505
 
 
'''
'''
prds = logreg.predict(X)
print metrics.confusion_matrix(y_test, y_pred_class)
'''
'''
confusion = matrix

sensetivity
1.0*confusion[1][1] /(confusion[1][0]+consfusion[1][1])

note: high number means it's good at picking up true positives

specificity
1.0*confusion[0][1]/ / (confusion[0][0]+confusion[0][1])

note: the ability to predict true nos

'''


# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!

df['bad_rating']=(df['stars']==1)|(df['stars']==2)
df.columns


##### Part 2 ######

# 1. Read in the titanic data set.

titanic = pd.read_csv('../data/titanic.csv')
titanic.head()
titanic.tail()
titanic.describe()
titanic.index
titanic.columns
titanic.shape
titanic.detail
titanic.dtypes

titanic.isnull().sum()

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1

titanic.SibSp
titanic[titanic['Name'].str.contains("Mrs.")]

titanic['wife']=(titanic['Name'].str.contains("Mrs.")&(titanic.SibSp >=1))

titanic[(titanic['Name'].str.contains('Mrs.') & (titanic['SibSp'] >=1 ))].count()

titanic[titanic.wife==True].count()
titanic[titanic.wife==False].count()

# 5. What is the average age of a male and
# the average age of a female on board?

titanic[titanic.Sex == 'male'].count()
titanic[titanic.Sex == 'female'].count()

avg_age_men = titanic.Age[titanic.Sex=='male'].mean()
avg_age_women = titanic.Age[titanic.Sex=='female'].mean()

avg_age_men
avg_age_women


# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages

titanic.isnull().sum()

titanic.Age[titanic.Sex=='male'] = titanic.Age[titanic.Sex=='male'].fillna(avg_age_men)


# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages

titanic.Age[titanic.Sex=='female'] = titanic.Age[titanic.Sex=='female'].fillna(avg_age_women)

# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors

#sns.pairplot(titanic, x_vars=['Age','wife'], y_vars='Survived', size=6, aspect=0.7)

featureslog = ['Age', 'wife']
X = titanic[featureslog]
y = titanic.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(featureslog, logreg.coef_[0])

'''
titanic.Age.fillna(titanic.Age.mean(), inplace=True)
feature_cols = ['Pclass', 'Parch', 'Age']
X = titanic[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg.fit(X_train, y_train)
zip(feature_cols, logreg.coef_[0])
y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)
'''

# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix

y_pred = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)


prds = logreg.predict(X)
print metrics.confusion_matrix(y_test, y_pred)

'''
[[125   3]
 [ 81  14]]

sensitivity = 14/95  =.184
specificity = 125/128  = .967

'''

# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!

#turn male and female into 1 and 0 values
titanic.Sex = titanic.Sex.replace(['male','female'],[1,0])

#split pclass into 1-3 columns 
pclass = pd.get_dummies(titanic.Pclass, prefix = 'Pclass')
titanic = pd.merge(titanic,pclass,left_index=True,right_index=True)



featureslog = ['Age', 'wife', 'Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fare']
X = titanic[featureslog]
y = titanic.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
zip(featureslog, logreg.coef_[0])



# 10. Show Accuracy, Sensitivity, Specificity

y_pred = logreg.predict(X_test)

#accuracy
print metrics.accuracy_score(y_test, y_pred)

#confusion matrix
print metrics.confusion_matrix(y_test, y_pred)
'''
[[111  17]
 [ 31  64]]
 
 
'''
confusion = metrics.confusion_matrix(y_test, y_pred)
#sensetivity
1.0*confusion[1][1] / (confusion[1][0]+confusion[1][1])
#specificity
1.0*confusion[0][0] / (confusion[0][0]+confusion[0][1])

# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

