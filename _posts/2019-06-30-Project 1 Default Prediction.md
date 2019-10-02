---
layout:     post
title:      Original-项目1 客户违约风险预测
subtitle:   Project 1 Prediction of customer default behavior
date:       2019-06-30
author:     Joe Zhang
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Python
    - Machine Learning
    - Classification
---

> 基于X公司提供的用户数据集，采用自建的GBDT和随机森林结合算法预测用户是否违约

# 数据清理 (Data Cleansing)

```python
import numpy as np
import pandas as pd
import csv
import gc #Garbage Collector,清理内存
from sklearn.preprocessing import LabelEncoder  #数据清洗_特征分拆
from sklearn.preprocessing import OneHotEncoder #数据清洗_特征分拆
from sklearn.preprocessing import Imputer #数据清洗_缺省值填充

# 读取数据
ClientData = pd.read_csv('OriginalData.csv')
CityGDPdf = pd.read_csv('CityGDPpc.csv')
SS300df = pd.read_csv('SS300.csv')

# 删除重复行
ClientData.drop_duplicates(inplace=True)

# 删除SERIALNO
ClientData.drop(['SERIALNO'],axis=1,inplace=True)

# 字符特征数值化
ClientData = ClientData.replace('无',0)
ClientData = ClientData.replace('有',1)

# 删除缺省值50%的值，无用features
ClientData = ClientData.replace('MISSING',-9999999)
ClientData = ClientData.replace(-8888888,-9999999)
B = pd.Series(((ClientData == -9999999).apply(sum) / (ClientData.shape[0])) > 0.5)
for feature in B.index:
    if B[feature] == True:
        ClientData.drop([feature],axis=1, inplace=True)
ClientData = ClientData.replace(-9999999, np.nan)

# 特征分解
ToSplit = ClientData.select_dtypes(include=[object]) #分离出需要进行特征转化的DataFrame：ToSplit
ClientData.drop(ToSplit.columns,axis=1,inplace=True)
ToSplit = ToSplit.fillna(method='pad')

# 文本类别替换成数值类别，如[南京，上海，成都]变成[0,1,2]
Le = LabelEncoder() 
ToSplit.drop(['PAYMENT_TYPE'],axis=1,inplace=True)
temp = pd.DataFrame()
for feature in ToSplit:
    value = Le.fit_transform(ToSplit[feature].values) 
    temp[feature] = pd.Series(value)
ToSplit = temp

# 数据分解，标签重置
for feature in ToSplit:
	enc = OneHotEncoder() #create a OneHotEncoder 创建独热向量编码
	df = pd.DataFrame(ToSplit[feature])
	enc.fit(df) #特征分解
	df = enc.transform(df).toarray()
	df = pd.DataFrame(df)
	a = list(range(df.shape[1]))
	b = list(map(lambda x:feature+"_"+str(x),a)) #新的feature名称是原来的名称加编号
	df.columns = b
	ToConcat = [df,ClientData]
	ClientData = pd.concat(ToConcat, axis=1)
	
# 缺省值替换
df = ClientData
imr = Imputer(missing_values='NaN', strategy='mean', axis=0) #均值填充缺失值
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
ClientData = pd.DataFrame(imputed_data,columns=ClientData.columns)

# 数据导出
ClientData.to_csv('CleanResultNoScrew00.csv', index=False)
print("Done!")
```

# 特征选择 (Feature Engineering)

```python
import numpy as np
import pandas as pd
import csv
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing

ClientData = pd.read_csv('CleanResultNoScrew00.csv')

# 特征缩放
min_max_scaler = preprocessing.MinMaxScaler()
ClientData = pd.DataFrame(min_max_scaler.fit_transform(ClientData),columns=ClientData.columns)

# 特征密度图
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

params = {
       'objective':"reg:logistic",
		'min_child_weight': 1,
        'eta': 0.2,
        'colsample_bytree': 0.5,
        'max_depth': 6,
        'subsample': 0.5,
        'alpha': 1,
        'gamma': 0,
        'silent': 0,
        'verbose_eval': False,
        'seed': 666
    }
rounds = 100

# XGBoost特征筛选
y = ClientData['TARGET_M4']
X = ClientData.drop(['TARGET_M4'],1)
xgtrain = xgb.DMatrix(X, label=y)
bst = xgb.train(params, xgtrain,num_boost_round=rounds)
  
# 创建Feature Map
features = [x for x in X.columns]
ceate_feature_map(features)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

# 将重要性输出至文件
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
df.to_csv("FeatureImportance00.csv", index = False)
importancefile = pd.read_csv("FeatureImportance00.csv")
importancefile.sort_values(by = "fscore", ascending=False,inplace=True)
importancefile.to_csv("FeatureImportance00.csv", index=False)

# 重要性绘图
plt.figure()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()

# 输出文件
ImportantFeature = ClientData[list(importancefile[importancefile['fscore']>0.01]['feature'].values)] # 去掉其余的features,只留下important的
ImportantFeature['Target'] = y
ImportantFeature.to_csv("ImportantFeature00.csv", index=False)
```

# 模型构建、训练与预测 (Building Model, fitting and predicting)

> 将GBDT输出节点记录（记录所在叶子的信息），拼接到原有的特征集中。最后新特征集输入进GBDT进行训练。

```python
import pandas as pd
import numpy as np
import gc
from sklearn import cross_validation,metrics, ensemble
from sklearn.cross_validation import train_test_split  # 该模块在0.18版本中被弃用，支持所有重构的类和函数都被移动到的model_selection模块。
from sklearn.ensemble import RandomForestClassifier # 引入随机森林
from sklearn.preprocessing import OneHotEncoder # 引入独热向量编码模块
from sklearn.ensemble import GradientBoostingRegressor # 引入GBDT模块
import datetime
starttime = datetime.datetime.now()

# 输入值是两个array或单列DataFrame
def CalF1Score(predict, actual):
	''' 计算F1 Score
	parameter
	---------
	predict: array
	预测值
	actual: array
	真实值
	'''
	TP = FN = FP = TN = 0
	for i in range(actual.shape[0]):
		if (actual[i] == predict[i]):
			if (actual[i] == 1):
				TP += 1
			else: 
				TN += 1
		else :
			if (actual[i] == 1):
				FN += 1
			else: 
				FP += 1
	return [TP,TN,FN,FP]	
	
ImportantFeature = pd.read_csv('ImportantFeature1.csv')
X = ImportantFeature[ImportantFeature.columns[:-1]]
y = ImportantFeature['Target']
	
# 构建新特征
TrainAccuracy = TestAccuracy = F1Score = AUC = KS = 0
F1ScoreTable = [0,0,0,0]
for ii in range(5):	
	# 随机划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

	# GBDT构造新特征
	gbr = ensemble.GradientBoostingClassifier() # 建立GBDT分类器
	from sklearn.grid_search import GridSearchCV
	parameter_grid = {'max_depth':[3,4,5],
					'n_estimators':[100],
					'learning_rate':[0.3,0.6,0.9],
					'loss':['deviance','exponential']}
	gridsearch = GridSearchCV(gbr, param_grid=parameter_grid, scoring='roc_auc',iid=True,cv=None)
	gridsearch.fit(X_train,y_train)
	best_param = gridsearch.best_params_

	gbr = ensemble.GradientBoostingClassifier(max_depth=best_param['max_depth'], learning_rate=best_param['learning_rate'], n_estimators=best_param['n_estimators'], loss=best_param['loss'], max_features=None) # 建立GBDT分类器
	gbr.fit(X_train,y_train) # 进行拟合
	
	enc = OneHotEncoder() # 创建独热编码
	enc.fit(gbr.apply(X_train)[:, :, 0]) 
	
	# 新特征拼接
	new_feature_train = pd.DataFrame(enc.transform(gbr.apply(X_train)[:, :, 0]).toarray(),index=X_train.index)
	FeaNum = int(new_feature_train.shape[1] / 10) #计算需要添加的特征数目，这里选取1取值总和大于总叶子数1/10的
	ix = new_feature_train.sum()[new_feature_train.sum() > FeaNum].index #选取所需叶子index
	new_feature_train = new_feature_train[new_feature_train.sum()[new_feature_train.sum() > FeaNum].index] #选取所需叶子index，并将其作为特征
	new_train = pd.concat([X_train,new_feature_train],axis = 1) #与原特征拼接
	del X_train, new_feature_train
	gc.collect()

	new_feature_test=pd.DataFrame(enc.transform(gbr.apply(X_test)[:, :, 0]).toarray(),index=X_test.index)
	new_feature_test=new_feature_test[ix] #选取所需叶子index，并将其作为特征
	new_test=pd.concat([X_test,new_feature_test],axis=1) #与原特征拼接
	del X_test, new_feature_test
	gc.collect()

	# 网格搜索最优参数
	rf = RandomForestClassifier()
	from sklearn.grid_search import GridSearchCV
	parameter_grid = {'n_estimators':[300,600],
					'max_depth':[6,8,10,None],
					'min_samples_split':[5,10,15]}
	gridsearch = GridSearchCV(rf, param_grid=parameter_grid,scoring='roc_auc',iid=True,cv=None)
	gridsearch.fit(new_train,y_train)
	best_param = gridsearch.best_params_

	rf = RandomForestClassifier(n_estimators=best_param['n_estimators'], criterion='entropy', max_depth=best_param['max_depth'], min_samples_split=best_param['min_samples_split']) # 创建随机森林分类器
	rf.fit(new_train,y_train) # 进行拟合

	# 训练集准确率
	output=rf.predict(new_train) # 测试集预测
	err = 0
	for i in range(new_train.shape[0]):
		if output[i] != y_train.iloc[i]:
			err = err +1
	err = err/new_train.shape[0]
	TrainAccuracy+=(1-err)*100

	# 测试集准确率
	output = rf.predict(new_test) # 测试集预测
	err = 0
	for i in range(new_test.shape[0]):
		if output[i] != y_test.iloc[i]:
			err = err +1
	err = err/new_test.shape[0]
	TestAccuracy += (1-err)*100

	F1ScoreTable = [F1ScoreTable[i]+CalF1Score(output,np.array(y_test))[i] for i in [0,1,2,3]] #F1值
	AUC += metrics.roc_auc_score(y_test,pd.DataFrame(rf.predict_proba(new_test))[1]) #AUC值
	fpr,tpr,threshold = metrics.roc_curve(y_test,pd.DataFrame(rf.predict_proba(new_test))[1])
	KS = KS + max(tpr - fpr) #K-S值
	
	print(ii)

# 输出精确度
print ("Logistic regression Train Accuracy :: %.6f%%"%(TrainAccuracy / 5))
print ("Logistic regression Test Accuracy :: %.6f%%"%(TestAccuracy / 5))

# 计算并输出F1 Score
PrecisionRate = (F1ScoreTable[0] + 1) / (F1ScoreTable[0] + F1ScoreTable[3]) 
RecallRate = (F1ScoreTable[0] + 1) / (F1ScoreTable[0] + F1ScoreTable[2])
arr = np.array([[F1ScoreTable[0], F1ScoreTable[2]], [F1ScoreTable[3], F1ScoreTable[1]]])
df = pd.DataFrame(arr)
df.index = ['Actual: 1','Actual: 0']
df.columns = [['Predict: 1','Predict: 0']]
print(df)
print("F1 Score: %10.9f" %(2 * PrecisionRate * RecallRate / (PrecisionRate + RecallRate))) 

print("AUC: %10.9f" %(AUC/5))
print("K-S: %10.9f" %(KS/5)) #输出K-S值

# 输出时间
endtime = datetime.datetime.now()
print ("Time: %10.3f"%((endtime - starttime).seconds))
```

