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