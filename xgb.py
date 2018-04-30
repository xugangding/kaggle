import pandas as pd 
from scipy import sparse as ssp 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
train=pd.read_csv('/opt/split/xaa')
test=pd.read_csv('/opt/data/test.csv')
train_label=train['is_attributed']
X=train.drop([''])
X_test=test.drop()