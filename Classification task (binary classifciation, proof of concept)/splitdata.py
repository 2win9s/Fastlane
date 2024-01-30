import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import math

# reading data from the link
SA = pd.read_csv('http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data')
SA.drop(['row.names','famhist'],axis=1, inplace = True) # famhist variable is removed
train_set_size = math.floor(len(SA)*0.75) # 75:25 split train:test
np.random.seed(863691977) # setting a seed for reproducible results
shuffled_indices = np.random.permutation(range(len(SA)))
SA_train = SA.iloc[shuffled_indices[:train_set_size],:].reset_index(drop=True)
SA_train.to_csv("SAheart.csv")
test = SA.iloc[shuffled_indices[train_set_size:],:].reset_index(drop=True)
test.to_csv("test.csv")
