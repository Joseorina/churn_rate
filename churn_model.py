## Churn Model
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('new_churn_data.csv')

## Data preparation
user_identifier = dataset['user']