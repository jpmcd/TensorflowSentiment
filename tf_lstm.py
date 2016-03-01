'''
TensorFlow implementation of Theano LSTM sentiment analyzer tutorial
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import tensorflow as tf
import tensorflow.python.ops.rnn_cell as rnn_cell

#import theano
#from theano import config
#import theano.tensor as tensor
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

class SentimentModel(object):

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._mask = tf.placeholder(tf.float32, [batch_size, num_steps])
        labels = tf.placeholder(tf.in32, [batch_size])
        mask = tf.transpose(self._mask)
        #mask = tf.expand_dims(tf.transpose(self._mask), -1)
        #mask_expand = tf.tile(mask, tf.pack([1, 1, size]))
        #self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        
        #################################
        
        #cell_output = tf.convert_to_tensor(cell_output)*mask_expand
        outputs = tf.convert_to_tensor(outputs)*mask
        mask_sum = tf.reduce_sum(mask, 0)
        proj = tf.reduce_sum(outputs, 0)/mask_sum #NOTE:did not tile mask_sum
        #proj.shape = [batch_size, size]
        #################################

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(proj, softmax_w) + softmax_b
        pred = tf.nn.softmax(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) 
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
            config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        

class Config(object):
    dim_proj=128  # word embeding dimension and LSTM number of hidden units.
    patience=10  # Number of epoch to wait before early stop if no progress
    max_epochs=500  # The maximum number of epoch to run
    dispFreq=10  # Display to stdout the training progress every N updates
    decay_c=0.  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
    vocab_size=10000  # Vocabulary size
    encoder='lstm'  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370  # Compute the validation error after this number of update.
    saveFreq=1110  # Save the parameters after every saveFreq updates
    maxlen=100  # Sequence longer then this get ignored
    batch_size=16  # The batch size during training.
    valid_batch_size=64  # The batch size used for validation/test set.
    num_layers=1
    dataset='imdb'  # Parameter for extra option
    noise_std=0.
    use_dropout=True  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None  # Path to a saved model we want to start from.
    test_size=-1  # If >0, we keep only this number of test example.

def get_config():
    return Config()

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype('float32')
    params = layers[options['encoder']][0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype('float32')
    params['b'] = numpy.zeros((options['ydim'],)).astype('float32')

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, vv in params.items():
        tparams[kk] = tf.Variable(vv)
    return tparams


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype('float32')

    return params

layers = {'lstm': (param_init_lstm, lstm_layer)}

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def build_model(tparams, options):
    
    # Used for dropout.
    use_noise = tf.Variable(0.)

    x = tf.placeholder

    


def main():

    tf.Graph().as_default()

    # Model options
    print("model options", model_options)

    load_data = imdb.load_data
    prepare_data = imdb.prepare_data

    print('Loading data')
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    tparams = init_tparams(params)





    session = tf.Session()

    session.close()

if __name__ == '__main__':
    tf.app.run()


