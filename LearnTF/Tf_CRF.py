# -*- coding: utf-8 -*-

"""
-------------------------------------------------
   File Name：     Tf_CRF
   Description :  tensorflow有关crf的学习笔记
   Author :       charl
   date：          2018/7/31
-------------------------------------------------
   Change Activity:
                   2018/7/31:
-------------------------------------------------
"""

import numpy as np
import tensorflow as tf

#data settings
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# 5 tags
# random features
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# 序列的长度
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

# Train and evaluate the model
with tf.Graph().as_default():
    with tf.Session() as sess:
        x_t = tf.constant(x)  # 观测序列
        y_t = tf.constant(y)  # 标记序列
        sequence_lengths_t = tf.constant(sequence_lengths)

        # weights shape = [100, 5]
        weights = tf.get_variable("weights", [num_features, num_tags])

        # matricized_x_t shape = [200, 100]
        matricized_x_t = tf.reshape(x_t, [-1, num_features])

        # 计算结果 [200, 5]
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)

        # unary_scores shape = [10, 20, 5]
        unary_scores = tf.reshape(matricized_unary_scores, [num_examples, num_words, num_tags])

        # compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        #   shape   shape   [10,20,5]   [10,20]   [10]
        # ======
        # 参数说明
        # inputs 一个形状为[batch_size, max_seg_len, num_tags]的tensor，一般使用BiLSTM处理之后输出转换为他要求的形状作为crf层的输入。
        # tag_indices： 一个形状为[batch_size, max_seq_len]的矩阵，其实就是真实标签
        # sequence_lengths: 一个形状为[batch_size] 的向量，表示每个序列的长度
        # transition_params: 形状为[num_tags, num_tags]的转移矩阵
        # return：
        #  log_likelihood: 标量,log-likelihood
        #  transition_params: 形状为[num_tags, num_tags] 的转移矩阵

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, y_t, sequence_lengths_t)

        # ==== viterbi_decode(score, transition_params) ===
        # 通俗一点，作用就是返回最好的标签序列，这个函数只能在测试时使用，在tensorflow外部解码
        # 参数说明：
        # score: 一个形状为[seq_len, num_tags]
        # tansition_params: 形状为[num_tags, num_tags]的转移矩阵
        # return
        # viterbi： 一个形状为[seq_len],显示了最高分的标签索引的列表
        # viterbi_score: A float containing the score for the Viterbi sequence.
        # ====================

        # ======= crf_decode ========
        # 在tensorflow内解码
        # 参数说明：
        #   potentials: 一个形状为[batch_size, max_seq_len, num_tags]的tensor
        # transition_params: 一个形状为[num_tags, num_tags]的转移矩阵
        # sequence_length: 一个形状为[batch_size] 的, 表示batch中每个序列的长度
        # return
        # decode_tags:一个形状为[batch_size, max_seq_len] 的tensor,类型是tf.int32.表示最好的序列标记.
        # best_score: 有个形状为[batch_size] 的tensor, 包含每个序列解码标签的分数.
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, sequence_lengths_t)
        # add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)

        # 定义梯度下降算法的优化器
        # learning_rate 0.01
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # train for a fixed number of iterations.
        sess.run(tf.global_variables_initializer())

        ''' 
       #eg:
       In [61]: m_20
       Out[61]: array([[ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])

       In [62]: n_20
       Out[62]: array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

       In [59]: n_20<m_20
       Out[59]: array([[ True,  True,  True,  True,  True,  True,  True,  True,  True, True]], dtype=bool)

        '''
        # 这里用mask过滤掉不符合的结果
        mask = (np.expand_dims(np.arange(num_words), axis=0) < np.expand_dims(sequence_lengths, axis=1))

        ###mask = array([[ True,  True,  True,  True,  True,  True,  True,  True,  True, True]], dtype=bool)
        # 序列的长度
        total_labels = np.sum(sequence_lengths)

        print("mask:", mask)

        print("total_labels:", total_labels)
        for i in range(1000):
            # tf_unary_scores,tf_transition_params,_ = session.run([unary_scores,transition_params,train_op])
            tf_viterbi_sequence, _ = sess.run([viterbi_sequence, train_op])
            if i % 100 == 0:
                '''
                false*false = false  false*true= false ture*true = true
                '''
                # 序列中预测对的个数
                correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)
