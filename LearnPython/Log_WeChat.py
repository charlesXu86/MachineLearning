# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     Log_WeChat
   Description :  wxpy操作微信
   Author :       charl
   date：          2018/7/31
-------------------------------------------------
   Change Activity:
                   2018/7/31:
-------------------------------------------------
"""

from wxpy import *

# 初始化机器人，扫码登陆
bot = Bot()
# 获取所有好友
my_friends = bot.friends()
print(type(my_friends))