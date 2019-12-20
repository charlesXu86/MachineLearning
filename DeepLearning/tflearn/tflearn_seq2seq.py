#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: tflearn_seq2seq.py 
@desc:  https://github.com/ichuang/tflearn_seq2seq/blob/master/tflearn_seq2seq.py
@time: 2017/10/23 
"""

from __future__ import division, print_function

import os
import sys
import tflearn
import argparse
import json

import numpy as np
import tensorflow as tf

from pattern import SequencePattern


class TFLearnSeq2Seq(object):
    '''
     seq2seq recurrent nerual network.
    '''
    AVAILABLE_MODELS = ["embedding_rnn", "embedding_attention"]

    def __init__(self, sequence_pattern, seq2seq_model=None, verbose=None, name=None, data_dir=None):
        self.sequence_pattern = sequence_pattern
        self.seq2seq_model = seq2seq_model or "embedding_rnn"
        assert self.seq2seq_model in self.AVAILABLE_MODELS
        self.in_seq_len = self.sequence_pattern.INPUT_SEQUENCE_LENGTH
        self.out_seq_len = self.sequence_pattern.OUTPUT_SEQUENCE_LENGTH
        self.in_max_int = self.sequence_pattern.INPUT_MAX_INT
        self.out_max_int = self.sequence_pattern.OUTPUT_MAX_INT
        self.verbose = verbose or 0
        self.n_input_symbols = self.in_max_int + 1
        self.n_output_symbols = self.out_max_int + 2  # extra one for GO symbol
        self.model_instance = None
        self.name = name
        self.data_dir = data_dir

    def generate_training_data(self, num_points):
        x_data = np.random.randint(0, self.in_max_int, size=(num_points, self.in_seq_len))
        x_data = x_data.astype(np.uint32)  # ensure integer type

        y_data = [ self.sequence_pattern.generate_output_sequence(x) for x in x_data]
        y_data = np.array(y_data)

        xy_data = np.append(x_data, y_data, axis=1)
        return xy_data, y_data

    def sequence_loss(self, y_pred, y_true):
        '''
          Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
          then use seq2seq.sequence_loss to actually compute the loss function.
        :param y_pred:
        :param y_true:
        :return:
        '''
        if self.verbose > 2 : print("my_sequence_loss y_pred=%s, y_true=%s" % (y_pred, y_true))
        logits = tf.unstack(y_pred, axis=1)
        targets = tf.unstack(y_true, axis=1)

    def CommandLine(args=None, arglist=None):
        '''
            Main command line.  Accepts args, to allow for simple unit testing.
            '''
        help_text = """
        Commands:
        train - give size of training set to use, as argument
        predict - give input sequence as argument (or specify inputs via --from-file <filename>)
        """

        parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

        parser.add_argument("cmd", help="command")
        parser.add_argument("cmd_input", nargs='*', help="input to command")
        parser.add_argument('-v', "--verbose", nargs=0,
                            help="increase output verbosity (add more -v to increase versbosity)", action=VAction, dest='verbose')
        parser.add_argument("-m", "--model",
                            help="seq2seq model name: either embedding_rnn (default) or embedding_attention", default=None)
        parser.add_argument("-r", "--learning-rate", type=float, help="learning rate (default 0.0001)", default=0.0001)
        parser.add_argument("-e", "--epochs", type=int, help="number of trainig epochs", default=10)
        parser.add_argument("-i", "--input-weights", type=str, help="tflearn file with network weights to load",
                            default=None)
        parser.add_argument("-o", "--output-weights", type=str,
                            help="new tflearn file where network weights are to be saved", default=None)
        parser.add_argument("-p", "--pattern-name", type=str, help="name of pattern to use for sequence", default=None)
        parser.add_argument("-n", "--name", type=str, help="name of model, used when generating default weights filenames",
                            default=None)
        parser.add_argument("--in-len", type=int, help="input sequence length (default 10)", default=None)
        parser.add_argument("--out-len", type=int, help="output sequence length (default 10)", default=None)
        parser.add_argument("--from-file", type=str, help="name of file to take input data sequences from (json format)",
                            default=None)
        parser.add_argument("--iter-num", type=int,
                            help="training iteration number; specify instead of input- or output-weights to use generated filenames",
                            default=None)
        parser.add_argument("--data-dir",
                            help="directory to use for storing checkpoints (also used when generating default weights filenames)",
                            default=None)
        # model parameters
        parser.add_argument("-L", "--num-layers", type=int, help="number of RNN layers to use in the model (default 1)",
                            default=1)
        parser.add_argument("--cell-size", type=int, help="size of RNN cell to use (default 32)", default=32)
        parser.add_argument("--cell-type", type=str, help="type of RNN cell to use (default BasicLSTMCell)",
                            default="BasicLSTMCell")
        parser.add_argument("--embedding-size", type=int, help="size of embedding to use (default 20)", default=20)
        parser.add_argument("--tensorboard-verbose", type=int, help="tensorboard verbosity level (default 0)", default=0)

        if not args:
            args = parser.parse_args(arglist)

        if args.iter_num is not None:
            args.input_weights = args.iter_num
            args.output_weights = args.iter_num + 1

        model_params = dict(num_layers = args.num_layers,
                            cell_size = args.cell_size,
                            cell_type = args.cell_type,
                            embedding_size = args.embedding_size,
                            learning_rate = args.learning_rate,
                            tensorboard_verbose = args.tensorboard_verbose)

        if args.cmd == "train":
            try:
                num_points = int(args.cmd_input[0])
            except:
                raise Exception("Please specify the number of datapoints to use for training, as the first argument")
            sp = SequencePattern(args.pattern_name, in_seq_len=args.in_len, out_seq_len=args.out_len)
            ts2s = TFLearnSeq2Seq(sp, seq2seq_model=args.model, data_dir=args.data_dir, name=args.name, verbose=args.verbose)
            ts2s.train(num_epochs=args.epochs, num_points=num_points, weights_output_fn=args.output_weights,
                   weights_input_fn=args.input_weights, model_params=model_params)
            return ts2s

        elif args.cmd == "predict":
            if args.from_file:
                inputs = json.loads(args.from_file)
            try:
                input_x = map(int, args.cmd_input)
                inputs = [input_x]
            except:
                raise Exception("Please provide a space-delimited input sequence as the argument")

            sp = SequencePattern(args.pattern_name, in_seq_len=args.in_len, out_seq_len=args.out_len)
            ts2s = TFLearnSeq2Seq(sp, seq2seq_model=args.model, data_dir=args.data_dir, name=args.name,
                                  verbose=args.verbose)
            results = []
            for x in inputs:
                prediction, y = ts2s.predict(x, weights_input_fn=args.input_weights, model_params=model_params)
                print("==> For input %s, prediction=%s (expected=%s)" % (x, prediction, sp.generate_output_sequence(x)))
                results.append([prediction, y])
            ts2s.prediction_results = results
            return ts2s

        else:
            print("Unknown command %s" % args.cmd)

    if __name__ == "__main__":
        CommandLine()