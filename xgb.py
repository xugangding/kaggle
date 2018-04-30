import pandas as pd 
from scipy import sparse as ssp 
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
train=pd.read_csv('/opt/split/xaa')
test=pd.read_csv('/opt/data/test.csv')
train_label=train['is_attributed']
feature_names=['app','device','os','channel']
X=train[feature_names]
X_test=test[feature_names]
for c in feature_names:
	le=LabelEncoder()
	le.fit(pd.concat([X,X_test])[c])
	X[c]=le.transform(X[c])
	X_test[c]=le.transform(X_test[c])

enc=OneHotEncoder()
enc.fit(pd.concat([X,X_test]))
X=enc.transform(X)
X_test=enc.transform(X_test)
for c in feature_names:
	d=train[c].value_counts().to_dict()
	train[c+"_count"]=train[c].apply(lambda x:d.get(x))
	dd=test[c].value_counts().to_dict()
	test[c+"_count"]=test[c].apply(lambda x:dd.get(x))
count_features=[c for c in train.columns.tolist() if ("count" in c)]
X=ssp.hstack(train[count_features].values,X).tocsr()
X_test=ssp.hstack(test[count_features].values,X_test)
XX_train, XX_test, y_train, y_test = train_test_split(X,train_label)
xgb_val=xgb.DMatrix(XX_test,y_test)
xgb_train=xgb.DMatrix(XX_train,y_train)
xgb_test=xgb.DMatrix(X_test)

