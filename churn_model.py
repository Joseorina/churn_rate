## Churn Model
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('new_churn_data.csv')

## Data preparation
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])

# One-hot encoding
dataset.housing.value_counts()
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns = ['housing_na','zodiac_sign_na', 'payment_type_na'])


# Splitting the dataset ontp Traiing Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'),
                                                    dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)

# Feature scaling and Balancing
y_train.value_counts()

pos_index = y_train[y_train == 1].index
neg_index = y_train[y_train == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    lower = pos_index
    higher = neg_index

random.seed(0)
higher = np.random.choice(higher, size = len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]


# Feature scaling
