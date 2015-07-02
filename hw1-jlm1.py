'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 7/1/2015
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('hw/data/police-killings.csv')
killings.head()
killings.tail()
killings.describe()
killings.index
killings.columns
killings.shape


# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race

killings.rename(columns={'lawenforcementagency':'agency', 'raceethnicity':'race'}, inplace=True)
killings.columns

# 2. Show the count of missing values in each column

killings.isnull()  
killings.isnull().sum()

killings[killings.streetaddress.isnull()]

# 3. replace each null value in the dataframe with the string "Unknown"

killings.streetaddress.fillna(value='Unknown', inplace=True)

# 4. How many killings were there so far in 2015?

killings[killings.year == 2015]

# 5. Of all killings, how many were male and how many female?

killings[killings.gender == 'Male'].count()
killings[killings.gender == 'Female'].count()

# 6. How many killings were of unarmed people?

killings[killings.armed == 'No']

# 7. What percentage of all killings were unarmed?

unarmed = killings[killings.armed == 'No'].count()
total = killings.count()

unarmed/total

#ufo.State.value_counts() / ufo.shape[0]

# 8. What are the 5 states with the most killings?

#drinks.groupby('continent').beer_servings.mean()

kbs = killings.groupby('state').count()
kbs.sort_index(by='city', ascending=False, inplace=True)
kbs.head(5)

# 9. Show a value counts of deaths for each race

killings.race.value_counts()


# 10. Display a histogram of ages of all killings

agecount = killings.groupby('age').count()
agecount.plot(kind='bar')

killings['age'].value_counts().plot(kind='bar')

# 11. Show 6 histograms of ages by race

killings.age.hist(by=killings.race, sharex=True, sharey=True)


# 12. What is the average age of death by race?

killings.groupby('race').age.mean()

# 13. Show a bar chart with counts of deaths every month

killings.groupby('month').count()

killings['month'].value_counts().plot(kind='bar')

###################
### Less Morbid ###
###################

majors = pd.read_csv('hw/data/college-majors.csv')
majors.head()
majors.tail()
majors.describe()
majors.index
majors.columns
majors.shape

# 1. Delete the columns (employed_full_time_year_round, major_code)

majors.drop(['Employed_full_time_year_round', 'Major_code'], axis=1, inplace = True)

# 2. Show the cout of missing values in each column

majors.isnull()  
majors.isnull().sum()

# 3. What are the top 10 highest paying majors?

majors.groupby('Major').count()
majors[['Major', 'Median']].sort_index(by='Median', ascending=False).head(10)


# 4. Plot the data from the last question in a bar chart, include proper title, and labels!

top10 = majors[['Major', 'Median']].sort_index(by='Median', ascending=False).head(10)
top10.plot(x='Major', y='Median', kind='bar', title='Highest Paid Majors')


# 5. What is the average median salary for each major category?

majors.groupby('Major_category').count()
majors.groupby('Major_category').Median.mean()

# 6. Show only the top 5 paying major categories

majors.groupby('Major_category', as_index=False).Median.mean().sort_index(by='Median', ascending=False).head(5)

# 7. Plot a histogram of the distribution of median salaries
'''
Todo...
'''
distsal = majors.groupby('Major').Median
distMed = majors['Median']
distMed.plot(kind='hist')

majors['Median'].hist(bins=20)


# 8. Plot a histogram of the distribution of median salaries by major category
#8 being Create a bar chart showing average median salaries for each major_category

majcatsal = majors.groupby('Major_category').Median.mean()
majcatsal.plot(x='Major_category', y='Median', kind='bar', title='Average Salary by Major Category')


# 9. What are the top 10 most UNemployed majors?
# What are the unemployment rates?

majors[['Major', 'Unemployment_rate']].sort_index(by='Unemployment_rate', ascending=False).head(10)
majors[['Major', 'Unemployment_rate']]


# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
# What are the unemployment rates?

majors.groupby('Major_category', as_index=False).Unemployment_rate.mean().sort_index(by='Unemployment_rate', ascending=False).head(10)
majors.groupby('Major_category').Unemployment_rate.mean()


# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042

majors['sample_employment_rate'] = majors.Employed / majors.Total
majors.head()

# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"

majors['sample_unemployement_rate'] = 1 - majors.sample_employment_rate
majors.head()
