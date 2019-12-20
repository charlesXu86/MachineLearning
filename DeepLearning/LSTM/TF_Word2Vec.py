#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: TF_Word2Vec.py 
@desc: tf实现word2vec
@time: 2018/04/25 
"""

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request as req
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'


# 下载文件
def maybe_download(filename, excepted_bytes):
    if not os.path.exists(filename):
        filename, _ = req.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)      # 用于在给定的路径上执行一个系统stat的调用
    if statinfo.st_size == excepted_bytes:
        print('Found and varified' + filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to varify ' + filename + '. Can you get it with browser?'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)

# 解压文件
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()  # 将数据转换成单词的列表
    return data

words = read_data(filename)
print('Data size:', len(words))

# 创建vocabulary 词汇表
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 反转形式
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)

del  words
print('Most common words (+ unk)', count[:5])
print('Sample Data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0 # 单词序号
def generate_batch(batch_size, num_skips, skip_window):
    '''
    生成训练用的batch数据
    :param batch_size:
    :param num_skips: 对每个单词生成多少个样本
    :param skip_indo: 单词最远可以联系的距离
    :return:
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # 初始化为数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # 定义span为对某个单词创建相关样本时用到的单词数量，包括前后和自己
    span = 2 * skip_window + 1
    # 创建一个双向队列,在对deque方法添加变量时，只会保留最后插入的span个变量
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # buffer中第skip_window个变量为目标单词
            target_to_avoid = [skip_window]

            # 对每次循环中对一个语境单词生成一个样本，先产生随机数，
            for j in range(num_skips):
                while target in target_to_avoid:
                    target = random.randint(0, span - 1)
                target_to_avoid.append(target)

                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2   # 每个目标单词提取的样本数

valid_size = 16  # 用来抽取的验证单词数
valid_window = 100  # 指验证单词只从频数最高的100个词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64   # 用来做负样本的噪声单词的数量

# 定义 skip-gram的word2vec 网络结构
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, -1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # 寻找train_inputs 对应的向量embed

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev= 1.0 / math.sqrt(embedding_size))
        )
        nce_bias = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights = nce_weights,
        biases = nce_bias,
        labels = train_labels,
        inputs = embed,
        num_sampled = num_sampled,
        num_classes = vocabulary_size
    ))

    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 计算向量embedding的L2范数
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

    # 得到标准化后
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 计算相似性
    '''
    参数：
        a: 一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量。
        b: 一个类型跟张量a相同的张量。
        transpose_a: 如果为真, a则在进行乘法计算前进行转置。
        transpose_b: 如果为真, b则在进行乘法计算前进行转置。
        adjoint_a: 如果为真, a则在进行乘法计算前进行共轭和转置。
        adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置。
        a_is_sparse: 如果为真, a会被处理为稀疏矩阵。
        b_is_sparse: 如果为真, b会被处理为稀疏矩阵。
        name: 操作的名字（可选参数）
    '''

    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as sess:
    init.run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000






