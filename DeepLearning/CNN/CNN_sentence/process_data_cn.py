#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: process_data_cn.py 
@desc: CNN 文本分词之中文预处理
@time: 2017/09/27 
"""

"""
  分词提取实词后，词汇量仍然相对较大
"""

import jieba
import jieba.posseg as pseg
import os

flag_list = ['t','q','p','u','e','y','o','w','m']
def jiebafenci(all_the_text):
    re = ""
    relist = ""
    words = pseg.cut(all_the_text)
    count = 0
    for w in words:
        flag = w.flag
        tmp = w.word
        if len(tmp)>1 and len(flag)>0 and flag[0] not in flag_list and tmp[0]>=u'/u4e00' and tmp[0]<=u'\u9fa5':
            re = re + " " + w.word
            count = count + 1
        if count%100 == 0:
            print(re)
            re = re.replace("\n", " ")
            relist = relist + "\n" + re
            re = ""
            count = count + 1
    re = re.replace("\n", " ").replace("\r", " ")
    if len(relist)>1 and len(re)>40:
        relist = relist + "\n" + re
    elif len(re)>40:
        relist = re
    relist = relist + "\n"
    relist = relist.replace("\r\n", "\n").replace("\n\n", "\n")

    return relist



def get_trainData(inpath, outfile):
    fw = open(outfile, 'a')
    for filename in os.listdir(inpath):
        print(filename)
        file_object = open(inpath + "\\" + outfile)
        try:
            all_the_text = file_object.read()
            all_the_text = all_the_text.decode("gb2312").encode("utf-8")
            pre_text = jiebafenci(all_the_text)
            if len(pre_text) > 30:
                fw.write(pre_text.encode("utf-8"))
        except:
            pass
        finally:
            file_object.close()
    fw.close()

    pass

inpath = ''
outfile = ''
get_trainData(inpath , outfile)