#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: SklearnPCA.py 
@desc: sklearn 做降维   pca
   主成分分析    pca
   因子分析      FactorAnalysis


@time: 2017/12/12 
"""

from sklearn import datasets
from sklearn import decomposition   # 导入分解模块


iris = datasets.load_iris()
iris_X = iris.data

# pca = decomposition.PCA()   # 初始化一个pca对象
pca = decomposition.PCA(n_components=2)
iris_X_prime = pca.fit_transform(iris_X)
a = iris_X_prime.shape
print(a)

# iris_pca = pca.fit_transform(iris_X)   #
# x = iris_pca[:5]
# y = pca.explained_variance_ratio_
# print(x, y)