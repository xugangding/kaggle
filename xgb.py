import pandas as pd 
import lightgbm as lgbm
from scipy import sparse as ssp 
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
train=pd.read_csv('/opt/split/xaa')
test=pd.read_csv('/opt/data/test.csv')
train_label=train['is_attributed']
train_test=pd.concat([train,test])
feature_names=['app','device','os','channel']
X=train[feature_names]
for c in feature_names:
	le=LabelEncoder()
	le.fit(train_test[c])
	train[c]=le.transform(train[c])
	test[c]=le.transform(test[c])

enc=OneHotEncoder()
enc.fit(train_test[feature_names])
X=enc.transform(train[feature_names])
X_test=enc.transform(test[feature_names])
for c in feature_names:
	d=train[c].value_counts().to_dict()
	train[c+"_count"]=train[c].apply(lambda x:d.get(x))
	dd=test[c].value_counts().to_dict()
	test[c+"_count"]=test[c].apply(lambda x:d.get(x))
count_features=[c for c in train.columns.tolist() if ("count" in c)]
X=ssp.hstack(train[count_features].values,X).tocsr()
X_test=ssp.hstack(test[count_features].values,X_test).tocsr()
XX_train, XX_test, y_train, y_test = train_test_split(X,train_label)
dtrain=lgbm.Dataset(XX_train,y_train)
dvalid=lgbm.Dataset(XX_test,y_test,reference=xgb_train)
learning_rate = 0.1
num_leaves = 15
min_data_in_leaf = 2000
feature_fraction = 0.6
num_boost_round = 10000
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": learning_rate,
          "num_leaves": num_leaves,
           "max_bin": 256,
          "feature_fraction": feature_fraction,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9
          }
bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, verbose_eval=100,early_stopping_rounds=100)
preds=bst.predict(X_test,num_iteration=bst.best_iteration)

