#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: process_data.py 
@desc:  https://github.com/LambdaWx/CNN_sentence_tensorflow/blob/master/process_data.py
@time: 2017/09/26 
"""

import numpy as np
import pickle   #py2是cpickle
from collections import defaultdict
import sys, re
import pandas as pd

vectot_size = 50
def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    数据二分类预处理
    Loads data and split into 10 folds
    :param data_folder: 
    :param cv: 
    :param clean_string: 
    :return: 
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    count = 0
    with open(pos_file, 'rb') as f:
        for line in f:
            if len(line) < 15:
                continue
            rev = []
            rev.append(line.strip())
            print(rev)
            if clean_string:
                """
                  Python中有join()和os.path.join()两个函数，具体作用如下：
                    join()：    连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
                    os.path.join()：  将多个路径组合后返回
                """
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,  #label
                     "text": orig_rev, # 原始文本
                     "num_words": len(orig_rev.split()), #该段文本的单词数量
                     "split":np.random.randint(0, cv)    #具体的cv值
            }
            revs.append(datum)
            count = count + 1
            if count > 200000:
                break

    count = 0
    with open(neg_file, 'rb') as f:
        for line in f:
            if len(line) < 15:
                continue
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
            count = count + 1
            if count > 200000:
                break
    return revs, vocab #返回原始的文本及对其的标记，单词表(word : word_count )

def build_data_cv_multi(data_folder, cv=10, clean_string=False):
    """
    多分类数据预处里
    :param data_folder: 
    :param cv: 
    :param clean_string: 
    :return: 
    """
    revs = []
    alt_athesim = data_folder[0]
    vocab = defaultdict(float)       #加一个默认值，详情见defaultdict的用法
    with open(alt_athesim, 'rb') as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                    "y":0,
                    "text":orig_rev,
                    "num_words": len(orig_rev.split()),
                    "split":np.random.randint(0, cv)
            }
            revs.append(datum)

    comp_graphics = data_folder[1]
    with open(comp_graphics, 'rb') as f:
        for line in f:
            rev = []
            rev.append(line.strip())   # strip（）用于一处字符串头尾指定的字符，( 默认为空格)
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                   "y":1,
                   "text": orig_rev,
                   "num_words": len(orig_rev.split()),
                   "split":np.random.randint(0, cv)
            }
            revs.append(datum)

    comp_os_ms_windows = data_folder[2]
    with open(comp_os_ms_windows, 'rb') as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {
                   "y": 2,
                   "text": orig_rev,
                   "num_words": len(orig_rev.split()),
                   "split": np.random.randint(0, cv)
            }
            revs.append(datum)
    return vocab, revs

def get_W(word_vecs, k=vectot_size):
    """
    Get word matrix, W[i] is the vector for word indexed by i 
    :param word_vecs: 
    :param k: 
    :return: 
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float64')
    W[0] = np.zeros(k, dtype='float64')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map  # W作为一个词向量矩阵，一个word可以通过word_idx_map得到其在W中的索引，进而得到其词向量

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    ''''
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float64').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float64')  
            else:
                f.read(binary_len)
    '''
    return word_vecs



def add_unknown_words(word_vecs, vocab, min_df=1, k=vectot_size):
    """
    For words that occur in at least min_df documents, create a seperate word vector
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    :param word_vecs: 
    :param vocab: 
    :param min_df: 
    :param k: 
    :return: 
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    :param String: 
    :param TREC: 
    :return: 
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    :param string: 
    :return: 
    """
    string = re.sub("[A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("\s{2,}", " ", string)
    return string.strip().lower()

if __name__ == "__main__":
    w2v_file = ""
    data_folder = ["E:\dataset\dataset_602151\corpus_6_4000\Auto_38.txt",
                   "E:\dataset\dataset_602151\corpus_6_4000\Culture_1278.txt"]
    print("Loading data...")
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)    #  ??????有错
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("Data Loaded!")
    print("Number of sentences:" + str(len(revs)))
    print("Vocab size:" + str(len(vectot_size)))
    print("Max sentence length:" + str(max_l))
    print("Loading word2vec vectors.....")
    w2v = load_bin_vec(w2v_file, vocab)
    print("Word2vec loaded!")
    print("Num words already in word2vec:" + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v) # 利用一个构建好的Word2vec向量来初始化词向量矩阵及词-向量映射表
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab) # 得到一个{word:word_vec}词典
    W2, _ = get_W(rand_vecs) #构建一个随机初始化的W2词向量矩阵
    pickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print("Dataset created!")
