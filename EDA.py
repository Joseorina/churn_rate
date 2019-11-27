import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('churn_data.csv')
"""
Exploratory Data Analysis
"""

dataset.head()
dataset.columns
dataset.describe()

dataset.isna().any()
dataset.isna().sum()
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])

## Histogram
dataset2 = dataset.drop(columns = ['user', 'churn'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histogram of numerical columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    
    vals = np.size(dataset2.iloc[:, i-1].unique())
    
    plt.hist(dataset2.iloc[:, i-1], bins=vals, color='blue')
plt.tight_layout(rect=[0,0.03,1,0.95])

## Piechart
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan','cancelled_loan','received_loan',
                    'rejected_loan','zodiac_sign','left_for_two_month_plus',
                    'left_for_one_month', 'is_referred']]

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    values = dataset2.iloc[:, i-1].value_counts(normalize=True).values
    index = dataset2.iloc[:,i-1].value_counts(normalize=True).index
    
    plt.pie(values, labels = index, autopct= '%1.1f%%')
    plt.axis('equal')
plt.tight_layout(rect=[0,0.03,1,0.95])

dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.cancelled_loan == 1].churn.value_counts()
dataset[dataset2.received_loan == 1].churn.value_counts()
dataset[dataset2.rejected_loan == 1].churn.value_counts()
dataset[dataset2.left_for_one_month == 1].churn.value_counts()

## Exploring Uneven Features
# Correlation Plot
dataset.drop(columns = ['churn', 'user', 'housing', 'payment_type',
                         'zodiac_sign']).corrwith(dataset.churn).plot.bar(
    figsize = (20, 10), title = 'Correlation with the Response variable',
    fontsize = 15, rot=45, grid = True)


## Correlation Matrix
sns.set(style = 'white')

# Computing the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()

# Generate a mask for upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True