#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: GBDT_Tree.py 
@desc: GBDT的决策树实现
@time: 2017/12/19 
"""

from math import log
from random import sample

class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value 的条件为< , 对于类别值的条件为=
        # 将满足条件的放入左树
        self.real_value_feature = None
        self.conditionValue = None
        self.leafNode = None

    def get_predict_value(self, instance):
        if self.leafNode:   # 到达叶子节点
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError('The Tree is null')
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"+str(self.conditionValue)+"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info

class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None

    def describe(self):
        return "{LeafNode:" + str(self.predictValue) + "}"

    def get_idset(self):
        return self.idset

    def get_predict_value(self):
        return self.get_predict_value()

    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_terminal_regions(targets, self.idset)

def MSE(values):
    '''
      计算均方误差
    :param values:
    :return:
    '''
    if len(values) < 2:
        return 0
    mean = sum(values)/float(len(values))
    error = 0.0
    for v in values:
        error += (mean - v) ** 2
    return error

def FriedmanMSE(left_values, right_values):
    # 假定每个样本的权重都为1
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_mean_left, total_mean_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_mean_left - total_mean_right
    return (weighted_n_left * weighted_n_right * diff * diff / (weighted_n_left + weighted_n_right))

def construct_desision_tree(dataset, remainedSet, targets, depth, leaf_nodes, max_depth, loss, criterion='MSE', split_points=0):
    if depth < max_depth:
        # 通过修改这里可以实现选择多少特征训练
        attributes = dataset.get_attributes()
        mse = -1
        selectedAttribute = None
        conditionValue = None
        selectedLeftIdSet = []
        selectedRightIdSet = []
        for attribute in attributes:
            is_real_type = dataset.is_real_type_field(attribute)
            attrValues = dataset.get_distinct_valueset(attribute)
            if is_real_type and split_points > 0 and len(attrValues) > split_points:
                attrValues = sample(attrValues, split_points)
            for attrValue in attrValues:
                leftIdSet = []
                rightIdSet = []
                for Id in remainedSet:
                    instance = dataset.get_instance(Id)
                    value = instance[attribute]
                    # 将满足条件的放入左子树
                    if (is_real_type and value < attrValue) or (not is_real_type and value == attrValue):
                        leftIdSet.append(Id)
                    else:
                        rightIdSet.append(Id)
                leftTargets = [targets[id] for id in leftIdSet]
                rightTargets = [targets[id] for id in rightIdSet]
                sum_mse = MSE(leftTargets) + MSE(rightTargets)
                if mse < 0 or sum_mse < mse:
                    selectedAttribute = attribute
                    conditionValue = attrValue
                    mse = sum_mse
                    selectedLeftIdSet = leftIdSet
                    selectedRightIdSet = rightIdSet
            if not selectedAttribute or mse < 0:
                raise ValueError("cannot determine the split attribute.")



