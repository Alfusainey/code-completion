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
def tokenize_space(words):
    words = words.split(" ")
    return words
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

training_data_token_array = read_data(training_file)

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
dictionary, reverse_dictionary = build_dataset(training_data_token_array)
vocab_size = len(dictionary)

test_data_tokens = "while (i <".split(" ")
n_input_test = len(test_data_tokens)
test_data = [dictionary[str(token)] for token in test_data_tokens]
test_data = np.reshape(np.array(test_data), [-1, n_input_test, 1])

# Parameters
learning_rate = 0.001
training_iters = 5000
display_step = 1000

# sequence_length
n_input = n_input_test

# units in RNN cell
n_hidden = 512

def vanilla_RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])    
    x = tf.split(x,n_input,1)    
    rnn_cell = rnn.BasicRNNCell(n_hidden)    
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    
def vanilla_LSTM(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])    
    x = tf.split(x,n_input,1)
        
    # 1-layer LSTM with n_hidden units but with lower accuracy.        
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)    
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def train_network(session, x, y):
    print("train_network")
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    
    writer.add_graph(session.graph)
    
    while step < training_iters:        
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data_token_array)-end_offset):
            offset = random.randint(0, n_input+1)        
        
        input_data = [ [dictionary[ str(training_data_token_array[i])]] for i in range(offset, offset+n_input) ]
        input_data = np.reshape(np.array(input_data), [-1, n_input, 1])
    
        target_onehot = np.zeros([vocab_size], dtype=float)
        target_onehot[dictionary[str(training_data_token_array[offset+n_input])]] = 1.0
        target_onehot = np.reshape(target_onehot,[1,-1])
    
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, prediction], feed_dict={x: input_data, y: target_onehot})
        loss_total += loss
        acc_total += acc
        
        if (step+1) % display_step == 0:
            #print("Iteration= " + str(step+1) + ", Average Loss= " + "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            input_context = [training_data_token_array[i] for i in range(offset, offset + n_input)]
            target_tokens = training_data_token_array[offset + n_input]
            predicted_token = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (input_context, target_tokens, predicted_token))
        step += 1
        offset += (n_input+1)
    print("Training Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))


# Test the input
def test_network(session, x, input_data):    
    onehot_pred = session.run(prediction, feed_dict={x: input_data})        
    top_k = tf.nn.top_k(onehot_pred, 5)
    top_k = session.run(top_k).values    
    predicted_tokens = []
    for v in top_k.ravel():
        try:
            predicted_tokens.append(reverse_dictionary[int(v)])
        except KeyError:
            continue
    print(predicted_tokens)

# Build Comptutational Graph
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
    
    prediction = vanilla_LSTM(x, weights, biases)
    
    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Model evaluation
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Launch the Session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train_network(session, x, y)
        print("Testing prediction")
        test_network(session, x, test_data)
        