#!/usr/bin/python
#coding: utf-8

from __future__ import division
import numpy as np
from MachineLearning.AdaBoost.weakclassify import WEAKC
from MachineLearning.utils import sign

class ADABC:
    def __init__(self,X,y,Weaker=WEAKC):
        '''
        :param X:
        :param y:
        :param Weaker:
        :return:
        '''
        self.X = np.array(X)
        # flatten 把数据撸平
        self.y = np.array(y).flatten(1)
        assert self.X.shape[1] == self.y.size
        self.Weaker = Weaker
        self.sums = np.zeros(self.y.shape)
        self.W = np.ones((self.X.shape[1],1)).flatten(1)/self.X.shape[1]
        self.Q = 0
    def train(self,M=4):
        '''
        :param M: M is the maximal Weaker classfication
        :return:
        '''
        self.G = {}
        self.alpha = {}
        for i in range(M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(M):
            self.G[i] = self.Weaker(self.X, self.y)
            e = self.G[i].train(self.W)
            self.alpha[i] = float(1.0 / 2 * np.log((1 - e) / e))
            sg = self.G[i].pred(self.X)
            Z = self.W * np.exp(-self.alpha[i] * self.y * sg.transpose())
            self.W = (Z/Z.sum()).flatten(1)
            self.Q = i
            print(self.finalclassifyer(i), '++++++++++++++++++++')

            if self.finalclassifyer(i) == 0:
                print(i+1, " weak classifier is enough to  make the error to 0")
                break

    def finalclassifyer(self,t):
        '''
        the 1 to t weak classifier come together
        :param t:
        :return:
        '''
        self.sums = self.sums + self.G[t].pred(self.X).flatten(1)* self.alpha[t]
        pre_y = sign(self.sums)
        t = (pre_y != self.y).sum()
        return t

    def pred(self,test_set):
        test_set = np.array(test_set)
        assert test_set.shape[0] == self.X.shape[0]
        sums = np.zeros((test_set.shape[1],1)).flatten(1)

        for i in range(self.Q + 1):
            sums = sums + self.G[i].pred(test_set).flatten(1)*self.alpha[i]
        pre_y = sign(sums)
        return pre_y


