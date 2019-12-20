#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: DataPro.py
@time: 2018/4/3 14:20
@desc: 数据处理
"""

import sys, pickle, os, random
import numpy as np

tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,  # 人名首字，人名非首字
             "B-LOC": 3, "I-LOC": 4,  # 地名首字，地名非首字
             "B-ORG": 5, "I-ORG": 6   # 组织机构名首字， 组织机构名非首字
             }

def read_corpus(corpus_path):
    '''
     read corpus and return the list of samples
    :param corpus_path:
    :return:
    '''
    data = []
    with open(corpus_path, encoding='utf-8') as f:
        lines = f.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data

def vocab_build(vocab_path, corpus_path, min_count):
    '''
    构建词向量
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    '''
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freg_words = []  # 低频词
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freg_words.append(word)
    for word in low_freg_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

def sentence2id(sent, word2id):
    '''
    句子转id
    :param sent:
    :param word2id:
    :return:
    '''
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('Vocab_size: ', len(word2id))
    return word2id

def random_embeding(vocab, embedding_dim):
    '''
    随机embedding
    :param vocab:
    :param embedding_dim:
    :return:
    '''
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

def pad_sequences(sequences, pad_mark=0):
    """
     如果目前序列长度参差不齐，这时需要使用pad_sequences将序列转化为经过填充以后的一个新序列
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    '''

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    '''
    if shuffle:
        random.shuffle(data) #把数据打乱

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels



