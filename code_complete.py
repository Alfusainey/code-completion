'''
AN LSTM implementation for code-completion using TensorFlow..
Next token prediction after n_input tokens learned from a source file.
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from scipy import spatial
import matplotlib.pyplot as plt


LOG_DIR = '/tmp/code_complete'
writer = tf.summary.FileWriter(LOG_DIR)

# Text file containing words for training
training_file = 'train.js'

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def tokenize(words):
    listoflists = [x.split("\n") for x in words] # split by space
    tokens = [val for sl in listoflists for val in sl if val]
    tokens = [val.strip() for val in tokens]
    
    listoflists = [t.split(".") for t in tokens]
    tokens = flatten(listoflists)
    listoflists = [v.split() for v in tokens]
    tokens = flatten(listoflists)
    return tokens

def flatten(list_of_lists):
    return [token for sublist in list_of_lists for token in sublist]

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = tokenize(content)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

training_data = read_data(training_file)

def build_dataset(words):
    #words = tuple([tuple(w) for w in words])
    #words = [element for t in words for element in t]
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)
print(vocab_size)
input_data = "while (i <"
input_data = tokenize(input_data)

# Parameters
learning_rate = 0.001
training_iters = 5000
display_step = 1000

# sequence_length
n_input = 4

# units in RNN cell
n_hidden = 512

with tf.Graph().as_default():
    # tf Graph input
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    
    def vanilla_RNN(x, weights, biases):
        x = tf.reshape(x, [-1, n_input])

        x = tf.split(x,n_input,1)

        rnn_cell = rnn.BasicRNNCell(n_hidden)

        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    def vanilla_LSTM(x, weights, biases):

        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x,n_input,1)

        # 1-layer LSTM with n_hidden units but with lower accuracy.        
        rnn_cell = rnn.BasicLSTMCell(n_hidden)
        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    pred = vanilla_RNN(x, weights, biases)
    
    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the Session
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)

            symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                print("Iteration= " + str(step+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/display_step))
                acc_total = 0
                loss_total = 0
                input_context = [training_data[i] for i in range(offset, offset + n_input)]
                target_tokens = training_data[offset + n_input]
                predicted_token = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (input_context, target_tokens, predicted_token))
            step += 1
            offset += (n_input+1)
        print("Training Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
