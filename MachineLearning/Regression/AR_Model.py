#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: AR_Model.py 
@desc: 自回归模型实现
     见博客：http://blog.csdn.net/zlzl8885/article/details/73188910?utm_source=itdadao&utm_medium=referral
            http://blog.csdn.net/shigangzwy/article/details/69525576
            http://blog.csdn.net/matrix_laboratory/article/details/53912312
@time: 2017/09/12 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

def date_parse(date):
    return pd.datetime.strftime(date, '%Y-%m')

if __name__ == '__main__':
    data = pd.read_csv('AirPassengers.csv', header=0, parse_dates=['Month'], date_parser=date_parse,
                       index_col=['Month'])
    p, d, q = 2, 1, 2
    data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    passengersNums = data['Passengers'].astype(np.float)
    logNums = np.log(passengersNums)
    subtractionNums = logNums - logNums.shift(periods=d)
    rollMeanNums = logNums.rolling(window=q).mean()
    logMRoll = logNums - rollMeanNums

    plt.plot(logNums, 'g-', lw=2, label=u'log of original')
    plt.plot(subtractionNums, 'y-', lw=2, label=u'subtractionNums')
    plt.plot(logMRoll, 'r-', lw=2, label=u'log of original - log of rollingMean')
    plt.legend(loc='best')
    plt.show()

    arima = ARIMA(endog=logNums, order=(p, d, q))
    proArima = arima.fit(disp=-1)
    fittedArima = proArima.fittedvalues.cumsum() + logNums[0]
    fittedNums = np.exp(fittedArima)
    plt.plot(passengersNums, 'g-', lw=2, label=u'orignal')
    plt.plot(fittedNums, 'r-', lw=2, label=u'fitted')
    plt.legend(loc='best')
    plt.show()
