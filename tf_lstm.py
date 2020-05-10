'''
A TensorFlow implementation of Theano LSTM sentiment analyzer tutorial,
this model is a variation on TensorFlow's ptb_word_lm.py seq2seq model
tutorial to accomplish the sentiment analysis task from the IMDB dataset.

'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import tensorflow as tf
import imdb


class SentimentModel(object):

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [num_steps, batch_size], name="inputs")
        self.mask = tf.placeholder(tf.float32, [num_steps, batch_size], name="mask")
        self.labels = tf.placeholder(tf.int64, [batch_size], name="labels")
        mask = tf.expand_dims(self.mask, -1)
        labels = self.labels
        #mask = tf.transpose(self._mask)
        #mask_expand = tf.tile(mask, tf.pack([1, 1, size]))
        #self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        #add LSTM cell and dropout nodes
        cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        #add dropout to input units
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[time_step, :, :], state)
                outputs.append(tf.expand_dims(cell_output, 0))
        
        outputs = tf.concat(0, outputs)*mask
        mask_sum = tf.reduce_sum(mask, 0)
        proj = tf.reduce_sum(outputs, 0)/mask_sum
        #NOW proj has shape [batch_size, size]

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(proj, softmax_w) + softmax_b
        pred = tf.nn.softmax(logits)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) 
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        correct_prediction = tf.equal(tf.argmax(pred,1), labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        if not is_training:
            return


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        #optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


class Config(object):
    patience=10  # Number of epoch to wait before early stop if no progress
    dispFreq=10  # Display to stdout the training progress every N updates
    decay_c=0.  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001  # Learning rate for sgd (not used for adadelta and rmsprop)
    vocab_size=10000  # Vocabulary size
    encoder='lstm'  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370  # Compute the validation error after this number of update.
    saveFreq=1110  # Save the parameters after every saveFreq updates
    maxlen=100  # Sequence longer then this get ignored
    batch_size=20  # The batch size during training.
    dataset='imdb'  # Parameter for extra option
    noise_std=0.
    use_dropout=True  # If False slightly faster, but worst test error. This frequently need a bigger model.
    reload_model=None  # Path to a saved model we want to start from.

    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 100
    hidden_size = 128
    max_epoch = 6
    max_max_epoch = 75
    keep_prob = 0.5
    lr_decay = 0.95


def get_minibatches_idx(n, batch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


def run_epoch(session, m, data, eval_op, verbose=False):
    print("batch size", m.batch_size)
    state = m.initial_state.eval()
    n_samples = data[0].shape[1]
    print("Testing %d samples:"%(n_samples))
   
    minibatches = get_minibatches_idx(n_samples, m.batch_size, shuffle=True)
    n_batches = len(minibatches)-1
    b_ind = 0
    correct = 0.
    total = 0

    for inds in minibatches[:-1]:
        print("\rbatch %d / %d"%(b_ind, n_batches), end="")
        sys.stdout.flush()

        x = data[0][:,inds]
        mask = data[1][:,inds]
        y = data[2][inds]

        cost, state, count, _ = session.run([m.cost, m.final_state, m.accuracy, eval_op],
                            {m.input_data: x, m.mask: mask, m.labels: y, m.initial_state: state})
        correct += count
        total += len(inds)
        b_ind += 1

    print("")
    accuracy = correct/total
    return accuracy


def get_config():
    return Config()


def main(unused_args):
    
    maxlen = 100
    n_words = 10000

    print('Loading data')
    train, valid, test = imdb.load_data(n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    train = imdb.prepare_data(train[0], train[1], maxlen=maxlen)
    valid = imdb.prepare_data(valid[0], valid[1], maxlen=maxlen)
    test = imdb.prepare_data(test[0], test[1], maxlen=maxlen)

    for data in [train, valid, test]:
        print(data[0].shape, data[1].shape, data[2].shape)

    config = get_config()
    eval_config = get_config()
    #eval_config.batch_size = 1
    #eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = SentimentModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse = True, initializer=initializer):
            mvalid = SentimentModel(is_training=False, config=config)
            mtest = SentimentModel(is_training=False, config=config)

        tf.initialize_all_variables().run()
        
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            start_time = time.time()
            train_acc = run_epoch(session, m, train, m.train_op) 
            print("Training Accuracy = %.4f, time = %.3f seconds\n"%(train_acc, time.time()-start_time))
            valid_acc = run_epoch(session, mvalid, valid, tf.no_op())
            print("Valid Accuracy = %.4f\n" % valid_acc)

        test_acc = run_epoch(session, mtest, test, tf.no_op())
        print("Test Accuracy = %.4f\n" % test_acc)


if __name__ == '__main__':
    tf.app.run()


