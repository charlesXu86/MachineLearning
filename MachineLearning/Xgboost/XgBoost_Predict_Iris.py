#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: XgBoost_Predict_Iris.py 
@desc: XgBoost对鸢尾花数据分类
@time: 2017/12/18 
"""

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # 做交叉验证用

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

if __name__=='__main__':
    path = 'F:\project\MachineLearning\MachineLearning\Data\Regression\iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)   # 交叉验证

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    bst = xgb.train(params, data_train, num_boost_round=50, evals=watch_list)
    y_hat = bst.predict(data_test)
    result = y_test.reshape(1, -1) == y_hat
    print('正确率: \t', float(np.sum(result)) / len(y_hat))
    print('End.... \n')