# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.eager.python import tfe

layers = tf.keras.layers


class RNN(tf.keras.Model):
  """A static RNN.

  Similar to tf.nn.static_rnn, implemented as a class.
  """

  def __init__(self, hidden_dim, num_layers, keep_ratio, forget_bias):
    super(RNN, self).__init__()
    self.keep_ratio = keep_ratio
    self.cells = tf.contrib.checkpoint.List([
        tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_dim,forget_bias=forget_bias)
        for _ in range(num_layers)
    ])

  def call(self, input_seq, training):
    batch_size = int(input_seq.shape[1])
    for c in self.cells:
      state = c.zero_state(batch_size, tf.float32)
      outputs = []
      input_seq = tf.unstack(input_seq, num=int(input_seq.shape[0]), axis=0)
      for inp in input_seq:
        output, state = c(inp, state)
        outputs.append(output)

      input_seq = tf.stack(outputs, axis=0)
      if training:
        input_seq = tf.nn.dropout(input_seq, self.keep_ratio)
    # Returning a list instead of a single tensor so that the line:
    # y = self.rnn(y, ...)[0]
    # in LSTMModel.call works for both this RNN and CudnnLSTM (which returns a
    # tuple (output, output_states).
    return [input_seq]


class Embedding(layers.Layer):
  """An Embedding layer."""

  def __init__(self, vocab_size, embedding_dim, **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

  def build(self, _):
    self.embedding = self.add_variable(
        "embedding_kernel",
        shape=[self.vocab_size, self.embedding_dim],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        trainable=True)

  def call(self, x):
    return tf.nn.embedding_lookup(self.embedding, x)


# pylint: disable=not-callable
class LSTMModel(tf.keras.Model):
  """LSTM for word language modeling.
  """

  def __init__(self,
               vocab_size,
               embedding_dim,
               hidden_dim,
               num_layers,
               dropout_ratio,
               use_cudnn_rnn=True,
               forget_bias=0.2):
    super(LSTMModel, self).__init__()

    self.keep_ratio = 1 - dropout_ratio
    self.use_cudnn_rnn = use_cudnn_rnn
    self.embedding = Embedding(vocab_size, embedding_dim)

    if self.use_cudnn_rnn:
      self.rnn = cudnn_rnn.CudnnLSTM(
          num_layers, hidden_dim, dropout=dropout_ratio)
    else:
      self.rnn = RNN(hidden_dim, num_layers, self.keep_ratio,forget_bias)

    self.linear = layers.Dense(
        vocab_size, kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
    self._output_shape = [-1, embedding_dim]

  def call(self, input_seq, training):
    """Run the forward pass of LSTMModel.

    Args:
      input_seq: [length, batch] shape int64 tensor.
      training: Is this a training call.
    Returns:
      outputs tensors of inference.
    """
    y = self.embedding(input_seq)
    if training:
      y = tf.nn.dropout(y, self.keep_ratio)
    y = self.rnn(y, training=training)[0]
    return self.linear(tf.reshape(y, self._output_shape))


def clip_gradients(grads_and_vars, clip_ratio):
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)


def loss_fn(model, inputs, targets, training):
  labels = tf.reshape(targets, [-1])
  outputs = model(inputs, training=training)
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=outputs))


def _divide_into_batches(data, batch_size):
  """Convert a sequence to a batch of sequences."""
  nbatch = data.shape[0] // batch_size
  data = data[:nbatch * batch_size]
  data = data.reshape(batch_size, -1).transpose()
  return data


def _get_batch(data, i, seq_len):
  slen = min(seq_len, data.shape[0] - 1 - i)
  inputs = data[i:i + slen, :]
  target = data[i + 1:i + 1 + slen, :]
  return tf.constant(inputs), tf.constant(target)


def evaluate(model, data):
  """evaluate an epoch."""
  total_loss = 0.0
  total_batches = 0
  start = time.time()
  for _, i in enumerate(range(0, data.shape[0] - 1, FLAGS.seq_len)):
    inp, target = _get_batch(data, i, FLAGS.seq_len)
    loss = loss_fn(model, inp, target, training=False)
    total_loss += loss.numpy()
    total_batches += 1
  time_in_ms = (time.time() - start) * 1000
  sys.stderr.write("eval loss %.2f (eval took %d ms)\n" %
                   (total_loss / total_batches, time_in_ms))
  return total_loss


def train(model, optimizer, train_data, sequence_length, clip_ratio):
  """training an epoch."""

  def model_loss(inputs, targets):
    return loss_fn(model, inputs, targets, training=True)

  grads = tfe.implicit_gradients(model_loss)

  total_time = 0
  for batch, i in enumerate(range(0, train_data.shape[0] - 1, sequence_length)):
    train_seq, train_target = _get_batch(train_data, i, sequence_length)
    start = time.time()
    optimizer.apply_gradients(
        clip_gradients(grads(train_seq, train_target), clip_ratio))
    total_time += (time.time() - start)
    if batch % 10 == 0:
      time_in_ms = (total_time * 1000) / (batch + 1)
      sys.stderr.write("batch %d: training loss %.2f, avg step time %d ms\n" %
                       (batch, model_loss(train_seq, train_target).numpy(),
                        time_in_ms))


class Datasets(object):

  def __init__(self, path):
    self.word2idx = {}
    self.idx2word = []  # integer id -> word string
    # Files represented as a list of integer ids (as opposed to list of string
    # words).
    self.word2idx = { (str(int(num/11**4)%11+1).zfill(2) + str(int(num/11**3)%11+1).zfill(2) + str(int(num/11**2)%11+1).zfill(2) + str(int(num/11)%11+1).zfill(2) +str(num%11+1).zfill(2)) : num for num in range(11*11*11*11*11) }
    if not FLAGS.predictpath :
      self.train = self.tokenize(os.path.join(path, "train.txt")) 
      self.valid = self.tokenize(os.path.join(path, "valid.txt"))
    else :
      self.predict = self.tokenize(os.path.join(path, "predict.txt")) 

  def get_key(self, value):
    return [k for k, v in self.word2idx.items() if v == value]

  def vocab_size(self):
    print("---------------vocab_size:%d\n" %(len(self.idx2word)))
    return 11**5


  def add(self, word):
    if word not in self.idx2word:
      self.idx2word.append(word)
      #print("append words:%s---------------vocab_size:%d\n" %(word,len(self.idx2word)))

  def tokenize(self, path):
    """Read text file in path and return a list of integer token ids."""
    tokens = 0
    with tf.gfile.Open(path, "r") as f:
      for line in f:
        words = line.strip()
        if(words!=''):
          tokens += 1
          self.add(words)

    # Tokenize file content
    with tf.gfile.Open(path, "r") as f:
      ids = np.zeros(tokens).astype(np.int64)
      token = 0
      for line in f:
        words = line.strip()
        if(words!=''):
            myword = self.word2idx.get(words)
            ids[token] = myword
            token += 1
    return ids


def predict(self):
  tf.enable_eager_execution()
  pre_data = Datasets(FLAGS.predictpath)
  sys.stderr.write( "pre_data.predice--------- :%s\n"% (pre_data.predict) )
  outputlist = []
  train_data = _divide_into_batches(pre_data.predict, 1)
  learning_rate = tf.contrib.eager.Variable(20.0, name="learning_rate")
  model = LSTMModel(
                     pre_data.vocab_size(),
                     FLAGS.embedding_dim,
                     FLAGS.hidden_dim, FLAGS.num_layers, 0,
                     False,0)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  checkpoint = tf.train.Checkpoint(
        learning_rate=learning_rate, model=model,
        optimizer=optimizer)

  checkpoint.restore(tf.train.latest_checkpoint(FLAGS.logdir))
  sys.stderr.write( "train_data--------------------------- :%s \n"% train_data )
  #inp, target = _get_batch(train_data, 0, FLAGS.seq_len)
  inputs = train_data[0:train_data.shape[0], :]
  inp = tf.constant(inputs)

  sys.stderr.write( "inp------------------- :%s \n"% inp )
  out = model(inp, training=False)
  sys.stderr.write( "out------------------- :%s \n"% out )
  pred_class_index=tf.argmax(out, 1,output_type=tf.int64).numpy()
  sys.stderr.write( "pred_class_index :%s \n"% pred_class_index )
  for i in pred_class_index:
    sys.stderr.write( "pred_num------------------index:%d- :%s \n"% (i,pre_data.get_key(i)) )

  a,b = out.shape
  lats_predict = out[a-1]
  sys.stderr.write( "out[%d]-------------- :%s \n"% (a,out[a-1]) )
  for i in range(3):
    max_operator=tf.argmax(lats_predict, 0,output_type=tf.int64).numpy()
    sys.stderr.write( "i th:%d-----operator: %d------nums :%s \n"% (i,max_operator,pre_data.get_key(max_operator)) )
    part1 = lats_predict[:max_operator]
    part2 = lats_predict[max_operator+1:]
    val = tf.constant([-1.])
    lats_predict = tf.concat([part1,val,part2], axis=0)


def main(_):
  tf.enable_eager_execution()

  if not FLAGS.data_path:
    raise ValueError("Must specify --data-path")
  corpus = Datasets(FLAGS.data_path)
  train_data = _divide_into_batches(corpus.train, FLAGS.batch_size)
  eval_data = _divide_into_batches(corpus.valid, 10)

  have_gpu = tfe.num_gpus() > 0
  use_cudnn_rnn = not FLAGS.no_use_cudnn_rnn and have_gpu

  with tf.device("/device:GPU:0" if have_gpu else None):
    # Make learning_rate a Variable so it can be included in the checkpoint
    # and we can resume training with the last saved learning_rate.
    learning_rate = tf.contrib.eager.Variable(20.0, name="learning_rate")
    model = LSTMModel(
                     corpus.vocab_size(),
                     FLAGS.embedding_dim,
                     FLAGS.hidden_dim, FLAGS.num_layers, FLAGS.dropout,
                     use_cudnn_rnn,0.2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    checkpoint = tf.train.Checkpoint(
        learning_rate=learning_rate, model=model,
        # GradientDescentOptimizer has no state to checkpoint, but noting it
        # here lets us swap in an optimizer that does.
        optimizer=optimizer)
    # Restore existing variables now (learning_rate), and restore new variables
    # on creation if a checkpoint exists.
    checkpoint.restore(tf.train.latest_checkpoint(FLAGS.logdir))
    sys.stderr.write("learning_rate=%f\n" % learning_rate.numpy())
    best_loss = None
    for _ in range(FLAGS.epoch):
      train(model, optimizer, train_data, FLAGS.seq_len, FLAGS.clip)
      eval_loss = evaluate(model, eval_data)
      if not best_loss or eval_loss < best_loss:
        if FLAGS.logdir:
          checkpoint.save(os.path.join(FLAGS.logdir, "ckpt"))
        best_loss = eval_loss
        '''
        sys.stderr.write( "model.variables:%s"% (model.variables))
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        model_path = "/tmp/tf/"
        save_path = saver.save(sess,model_path)
        '''
      else:
        learning_rate.assign(learning_rate / 4.0)
        sys.stderr.write("eval_loss did not reduce in this epoch, "
                         "changing learning rate to %f for the next epoch\n" %
                         learning_rate.numpy())
      sys.stderr.write( "one epoch,best_loss: :%f \n"% (best_loss))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data-path",
      type=str,
      default="")
  parser.add_argument(
      "--predictpath", type=str, default="", help="Directory for checkpoint.")
  parser.add_argument(
      "--logdir", type=str, default="/tmp/tf/", help="Directory for checkpoint.")
  parser.add_argument("--epoch", type=int, default=20, help="Number of epochs.")
  parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
  parser.add_argument(
      "--seq-len", type=int, default=8, help="Sequence length.")
  parser.add_argument(
      "--embedding-dim", type=int, default=50, help="Embedding dimension.")
  parser.add_argument(
      "--hidden-dim", type=int, default=50, help="Hidden layer dimension.")
  parser.add_argument(  
      "--num-layers", type=int, default=2, help="Number of RNN layers.")
  parser.add_argument(
      "--dropout", type=float, default=0.2, help="Drop out ratio.")
  parser.add_argument(
      "--clip", type=float, default=0.2, help="Gradient clipping ratio.")
  parser.add_argument(
      "--no-use-cudnn-rnn",
      action="store_true",
      default=False,
      help="Disable the fast CuDNN RNN (when no gpu)")

  FLAGS, unparsed = parser.parse_known_args()
  sys.stderr.write( "run FLAGS: %s   %s \n" %
                         (FLAGS,unparsed))
  if FLAGS.predictpath :
    sys.stderr.write( "predict mode: %s  , %s \n" %
                         (FLAGS,unparsed))
    tf.app.run(main=predict, argv=[sys.argv[0]] + unparsed)
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)