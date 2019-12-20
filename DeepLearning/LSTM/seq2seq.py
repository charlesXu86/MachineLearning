'''
Pedagogical example realization of seq2seq recurrent neural networks, using TensorFlow and TFLearn.
More info at https://github.com/ichuang/tflearn_seq2seq
'''
# ==========================
'''
  seq2seq 模型介绍
  seq2seq理论参考博客：http://blog.csdn.net/sunlylorn/article/details/50607376
'''

from __future__ import division, print_function

import os
import  sys
import tflearn
import argparse
import json

import numpy as np
import tensorflow as tf

from tensorflow.contrib import seq2seq
from tensorflow.contrib import rnn

# -------------------------------------------

class SequencePattern(object):

    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    INPUT_MAX_INT = 9
    OUTPUT_MAX_INT = 9
    PATTERN_NAME = "SORTED"

    def __init__(self, name=None, in_seq_len=None, out_seq_len=None):
        if name is not None:
            assert hasattr(self, "%s_sequence" % name)
            self.PATTERN_NAME = name
        if in_seq_len:
            self.INPUT_SEQUENCE_LENGTH = in_seq_len
        if out_seq_len:
            self.OUTPUT_SEQUENCE_LENGTH = out_seq_len

    def generate_output_sequence(self, x):
        '''
        For a given input sequence,generate the output_sequence
        :param x: a 1D numpy array
        :return: 1D numpy array of length OUTPUT_SEQUENCE_LENGTH
        '''
        return getattr(self, "%s_sequence" % self.PATTERN_NAME)(x)

    def maxmin_dup_sequence(self, x):
        '''
        Generate sequence with [max,min,rest of original entries]
        :param x: 
        :return: 
        '''
        x = np.array(x)
        y = [x.max(), x.min()] + list(x[2:])
        return np.array(y)[:self.OUTPUT_SEQUENCE_LENGTH] # truncate at out seq_len

    def sorted_sequence(self, x):
        '''
        Generate sorted version of original sequence
        :param x: 
        :return: 
        '''
        return np.array(sorted(x))[:self.OUTPUT_SEQUENCE_LENGTH]

    def reversed_sequence(self, x):
        '''
        Generate reversed version of original sequence
        :param x: 
        :return: 
        '''
        return np.array( x[::-1] )[:self.OUTPUT_SEQUENCE_LENGTH]

# -------------------------------------------------------------

class TFLearnSeq2Seq(object):
    '''
    seq2seq recurrent nurual network, implemented using TFLearn.
    '''
    AVAILBLE_MODELS = ["embedding_rnn", "embedding_attention"]
    def __init__(self, sequence_pattern, seq2seq_model=None, verbose=None, name=None, data_dir=None):
        '''
        
        :param sequence_pattern: 
        :param seq2seq_model: 
        :param verbose: 
        :param name: 
        :param data_dir: 
        '''
        self.sequence_pattern = sequence_pattern
        self.seq2seq_model = seq2seq_model or 'embedding_rnn'
        assert  self.seq2seq_model in self.AVAILBLE_MODELS
        self.in_seq_len = self.sequence_pattern.INPUT_SEQUENCE_LENGTH
        self.out_seq_len = self.sequence_pattern.OUTPUT_SEQUENCE_LENGTH
        self.in_max_len = self.sequence_pattern.INPUT_Max_INT
        self.out_max_int = self.sequence_pattern.OUTPUT_MAX_INT
        self.verbose = verbose or 0
        self.n_input_symbols = self.in_max_len + 1
        self.n_output_symbols = self.out_max_int + 2   # extro one for GO symbol
        self.model_instance = None
        self.name = name
        self.data_dir = data_dir

    def generate_training_data(self, num_points):
        '''
        
        :param num_points: 
        :return: 
        '''
        x_data = np.random.randint(0, self.in_max_int, size=(num_points, self.in_seq_len))  # shape [num_points, in_seq_len]
        x_data = x_data.astype(np.uint32)

        y_data = [ self.sequence_pattern.generate_output_sequence(x) for x in x_data]
        y_data = np.array(y_data)

        xy_data = np.append(x_data, y_data, axis=1)
        return xy_data, y_data

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN, Reshape prediected and true(label) tensors,generate dummy weights
        then use seq2seq.sequence_loss to actually compute the loss function.
        :param y_pred: 
        :param y_true: 
        :return: 
        '''
        if self.verbose > 2 : print("my_sequence_loss y_pred=%s, y_true=%s" % (y_pred, y_true))
        logits = tf.unpack(y_pred, axis=1)  # list of [-1,num_decoder_symbols] elements
        targets = tf.unpack(y_true,axis=1)  #
        if self.verbose > 2 :
            print("my_sequence_loss logits=%s" % (logits))
            print("my_sequence_loss targets=%s" % (targets))
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        if self.verbose > 4 : print("my_sequence_loss wights=%s" % (weights))
        s1 = seq2seq.sequence_loss(logits, targets, weights)
        if self.verbose > 2 : print("my_sequence_loss return %s" % s1)
        return  s1

    def accuracy(self, y_pred, y_true, x_in):
        '''
        Compute the accuracy of prediction based on the true labels.Use the average number of equal values
        :param y_pred: 
        :param y_true: 
        :param x_in: 
        :return: 
        '''
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))
        if self.verbose > 2 : print("my_accuracy pred_idx = %s" % pred_idx)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')
        return accuracy

    def model(self, mode="train", num_layers=1, cell_size=32, cell_type="BasicLSTMCell", embedding_size=20, learning_rate=0.0001,
              tensorboard_verbose=0,checkpoint_path=None):
        '''
        Build tensor specifying graph of oprations for the seq2seq neural network model
        :param mode: string, either "traim" or "predict"
        :param num_layers: number of RNN cell layers to use
        :param cell_size: size for hidden layer in the RNN cell
        :param cell_type: attribute of rnn_cell specifying which RNN cell type to use
        :param embedding_size:
        :param learning_rate:
        :param tensorboard_verbose:
        :param checkpoint_path:
        :return: TFLearn model instance. Use DNN model for this
        '''
        assert mode in ["train", "predict"]
        checkpoint_path = checkpoint_path or ("%s%ss2s_checkpoint.tfl" % (self.data_dir or "", "/" if self.data_dir else ""))
        GO_VALUE = self.out_max_int + 1   # unique integer value used to trigger decoder outputs in the seq2seq RNN

        network  = tflearn.input_data(shape=[None, self.in_seq_len + self.out_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(network, [0, 0], [-1, self.in_seq_len], name="enc_in")  # get encoder inputs
        encoder_inputs = tf.unpack(encoder_inputs, axis=1)    # Transform into list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(network, [0,0], [-1, self.out_seq_len], name="dec_in")
        decoder_inputs = tf.unpack(decoder_inputs, axis=1)

        go_input = tf.mul(tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE)
        decoder_inputs = [go_input] + decoder_inputs[: self.out_seq_len-1]

        feed_previous = not (mode=="train")

        if self.verbose > 3:
            print("feed_previous = %s" % str(feed_previous))
            print("ecoder inputs: %s" % str(encoder_inputs))
            print("decoder input %s" % str(decoder_inputs))
            print("len decoder inputs: %s" % len(decoder_inputs))

        self.n_input_symbols = self.in_max_len + 1   # default is integers from 0 to 9
        self.n_output_symbols = self.out_max_int + 2 # extra "GO" symbol for decoder inputs

        single_cell = getattr(rnn, cell_type)(cell_size, state_is_tuple=True)
        if num_layers == 1:
            cell = single_cell
        else:
            cell = rnn.MultiRNNCell([single_cell] * num_layers)

        if self.seq2seq_model == "embedding_rnn":
            model_outputs, states = seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                  decoder_inputs,
                                                                  cell,
                                                                  num_encoder_symbols=self.n_input_symbols,
                                                                  num_decoder_symbols=self.n_output_symbols,
                                                                  embedding_size=embedding_size,
                                                                  feed_previous=feed_previous)
        elif self.seq2seq_model=="embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                        decoder_inputs,
                                                                        cell,
                                                                        num_encoder_symbols=self.n_input_symbols,
                                                                        num_decoder_symbols=self.n_output_symbols,
                                                                        embedding_size=embedding_size,
                                                                        num_heads=1,
                                                                        initial_state_attention=False,
                                                                        feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        # For TFLearn to know what to save and restore
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model", model_outputs)

        if self.verbose > 2: print("packed model outputs: %s" % network)
        if self.verbose > 3:
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
            print("all_vars = %s" % all_vars)

        with tf.name_scope("TargetData"):
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        network = tflearn.regression(network,
                                     placeholder='adam',
                                     learning_rate=learning_rate,
                                     loss=self.sequence_loss,
                                     metric=self.accuracy,
                                     name="Y")

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose, checkpoint_path=checkpoint_path)
        return model

    def train(self, num_epochs=20, num_points=100000, model=None, model_params=None, weights_input_fn=None,
              validation_set=0.1, snapshot_step=5000, batch_size=128,weight_output_fn=None):
        '''
        Train model, with specified number of epochs, and dataset size.
        Use specified model, or create one if not provided.  Load initial weights from file weights_input_fn,
        if provided. validation_set specifies what to use for the validation.

        :param num_epochs:
        :param num_points:
        :param model:
        :param model_params:
        :param weights_input_fn:
        :param validation_set:
        :param snapshot_step:
        :param batch_size:
        :param weight_output_fn:
        :return:  Returns logits for prediction, as an numpy array of shape [out_seq_len, n_output_symbols].
        '''
        trainXY, trainY = self.generate_training_data(num_points)
        print("[TFLearnSeq2seq] Training on %d point dataset (pattern '%s'),with %d epochs" % (num_points,
                                                                                               self.sequence_pattern.PATTERN_NAME,
                                                                                               num_epochs))
        if self.verbose > 1:
            print("  model parameters: %s" % json.dumps(model_params, indent=4))
        model_params = model_params or {}
        model = model or self.setup_model("train", model_params, weights_input_fn)

        model.fit(trainXY, trainY,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  show_metric=True,
                  snapshot_step=snapshot_step,
                  snapshot_epoch=False,
                  run_id="TFLearnSeq2Seq"
                  )
        print("Done")
        if weight_output_fn is not None:
            weight_output_fn = self.conno

    def canonical_weights_fn(self, iteration_num=0):
        '''
        Construct canonical weights filename,based on model and pattern names
        :param iteration_num:
        :return:
        '''
        if not type(iteration_num) == int:
            try:
                iteration_num = int(iteration_num)
            except Exception as err:
                return iteration_num
        model_name = self.name or "basic"
        wfn = "ts2s__%s__%s_%s.tfl" % (model_name, self.sequence_pattern.PATTERN_NAME, iteration_num)
        if self.data_dir:
            wfn = "%s/%s" % (self.data_dir,wfn)
        self.weights_filename = wfn
        return wfn

    def setup_model(self, mode, model_params=None, weights_input_fn=None):
        '''
        Setup a model instance, using the specified mode and model parameters.
        Load the weights from the specified file, if it exists.
        If weights_input_fn is an integer, use that the model name, and
        the pattern name, to construct a canonical filename.
        :param mode:
        :param model_params:
        :param weghts_input_fn:
        :return:
        '''
        model_params = model_params or {}
        model = self.model_instance or self.model(mode=mode, **model_params)
        self.model_instance = model
        if weights_input_fn:
            if type(weights_input_fn)==int:
                weghts_input_fn = self.canonical_weights_fn(weights_input_fn)
            if os.path.exists(weights_input_fn):
                model.load(weghts_input_fn)
                print("[TFLearnSeq2Seq] model weights loaded from %s" % weights_input_fn)
            else:
                print("[TFLearnSeq2Seq] MISSING model weights file %s" % weights_input_fn)
        return model

    def predict(self, Xin, model=None, model_params=None, weights_input_fn=None):
        '''
        Make a prediction, using the seq2seq model, for the given input sequence Xin.
        :param Xin:
        :param model:
        :param model_params:
        :param weights_input_fn:
        :return: prediction, y
        '''

