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
