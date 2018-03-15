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
from collections import Counter

LOG_DIR = '/tmp/code_complete'
writer = tf.summary.FileWriter(LOG_DIR)

# Text file containing words for training
training_file = 'file.js'
#training_file = 'train.js'

# add the most common tokens to the dictionary
most_common_counter = 700

# number of lines of code we want to train with
train_lines = 30000

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
    words2 = []
    for word in words:
        word = str(word).replace(")", " ) ").replace("(", " ( ").replace(".", " . ").replace(";", " ; ").replace("\"", " \" ").replace(",", " , ")
        words2.append(word)
    words = words2
    
    listoflists = [x.split("\n") for x in words] # split by space
    
    tokens = [val for sl in listoflists for val in sl if val]
    tokens = [val.strip() for val in tokens]
    tokens = flatten(listoflists)
    
    listoflists = [v.split() for v in tokens]
    
    tokens = flatten(listoflists)
    
    cnt = Counter(tokens)
    most_comm = []
    for count in cnt.most_common(most_common_counter):
        most_comm.append(count[0])

    tokens2 = []
    for token in tokens:
        if token in most_comm:
            tokens2.append(token)
        else:
            tokens2.append("EMPTY")

    tokens = tokens2
    return tokens

def flatten(list_of_lists):
    return [token for sublist in list_of_lists for token in sublist]

def read_data(fname):
    with open(fname, 'rb') as f:
        #f.encode('utf-8') #fuer grosses file
        content = f.readlines()[0:train_lines] #30000
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

# Parameters
learning_rate = 0.001
training_iters = 1000 #5000
display_step = 1000 #1000

# sequence_length
n_input = 4

# units in RNN cell
n_hidden = 512


def vanilla_RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])    
    x = tf.split(x,n_input,1)    
    rnn_cell = rnn.BasicRNNCell(n_hidden)    
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases
    
def vanilla_LSTM(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])    
    x = tf.split(x,n_input,1)
        
    # 1-layer LSTM with n_hidden units but with lower accuracy.        
    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)    
    return tf.matmul(outputs[-1], weights) + biases



# config for using cpu
config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )

def train_network(session, prediction, x, y, cost, optimizer, accuracy):
            
    # Initialize some variables
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    #writer.add_graph(session.graph)
    
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
            print("Iteration= " + str(step+1) + ", Average Loss= " + "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            input_context = [training_data_token_array[i] for i in range(offset, offset + n_input)]
            target_tokens = training_data_token_array[offset + n_input]
            predicted_tokens = top_k(session, onehot_pred)
            print("%s - [%s] vs [%s]" % (input_context, target_tokens, predicted_tokens))
        step += 1
        offset += (n_input+1)
    print("Training Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))

# return the top_k predictions of the input
def top_k(session, raw_predictions, k=5):
    predictions = []
    top_k = tf.nn.top_k(raw_predictions, k)
    top_k = session.run(top_k).values
   
   
    numbers = []
    for number in top_k.ravel():
        number = int(number + 0.5)
        if number not in numbers:
            numbers.append(number)
    for v in numbers:
        try:
            predictions.append(reverse_dictionary[int(v)])
        except KeyError: # unknown token
            continue
   
    return predictions

def do_prediction(session, input_data, x, prediction):
    # TEST
    input_data = [input_data]
    test_data_tokens = tokenize(input_data)
    test_data_tokens = test_data_tokens[-n_input:]

    # test_data_tokens = input_data.split(" ") #CodeCompleteGUI.get_input() #"while (i <".split(" ")
    n_input_test = len(test_data_tokens)
    
    test_data = []
    for token in test_data_tokens:
        try:
            test_data.append(dictionary[str(token)])
        except KeyError:
            test_data.append(dictionary[str("EMPTY")])
            continue
    
    test_data = np.reshape(np.array(test_data), [-1, n_input_test, 1])
    session.run(tf.global_variables_initializer())
    onehot_pred = session.run(prediction, feed_dict={x: test_data})
    return top_k(session, onehot_pred)


def predict(input_data):
    path = "/home/alfu/Desktop/savedModel/"  # TODO: Make this configurable
    tf.reset_default_graph()
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(path + "code_complete_model.meta")
        saver.restore(session, tf.train.latest_checkpoint(path))

        graph = tf.get_default_graph()
        weights = graph.get_tensor_by_name('weights:0')
        biases = graph.get_tensor_by_name("biases:0")
        x = graph.get_tensor_by_name("x:0")
        prediction = vanilla_LSTM(x, weights, biases)
        return do_prediction(session, input_data, x, prediction)


def save_model(session, path):
    saver = tf.train.Saver()
    savePath = saver.save(session, path)
    return savePath


# Build Comptutational Graph
def build_graph():
    with tf.Graph().as_default():
        # tf Graph input
        x = tf.placeholder("float", [None, n_input, 1], name="x")
        y = tf.placeholder("float", [None, vocab_size], name="y")

        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]), name="weights")
        }
        biases = {
            'out': tf.Variable(tf.random_normal([vocab_size]), name="biases")
        }

        prediction = vanilla_LSTM(x, weights['out'], biases['out'])

        # Loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        # Model evaluation
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Launch the Session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            train_network(session, prediction, x, y, cost, optimizer, accuracy)
            print("Saving the model")
            p = save_model(session, '/home/raphael/Desktop/savedModel/code_complete_model')
            print("Model save in the file: " + p)
            # results = test_network(session, cc.get_input())
            # cc.setOutput(results)


if __name__ == '__main__':
    # Run this file to train the network
    build_graph()

"""
# Test the input
def test_network(session, input_data, x, prediction):

    #TEST
    input_data = [input_data]
    test_data_tokens = tokenize(input_data)
    test_data_tokens = test_data_tokens[-n_input:]

    #test_data_tokens = input_data.split(" ") #CodeCompleteGUI.get_input() #"while (i <".split(" ")
    n_input_test = len(test_data_tokens)
    print(test_data_tokens)
    test_data = []
    for token in test_data_tokens:
        try:
            test_data.append(dictionary[str(token)])
        except KeyError:
            test_data.append(dictionary[str("EMPTY")])
            continue
    print (test_data)
    test_data = np.reshape(np.array(test_data), [-1, n_input_test, 1])
    onehot_pred = session.run(prediction, feed_dict={x: test_data})
    return top_k(session, onehot_pred) #gui.setOutput(predicted_tokens)
def save_model(session, path):
    saver = tf.train.Saver(max_to_keep=1)
    savePath = saver.save(session, path)
    return savePath
def restore_model(session, path, feed_dict):
    saver = tf.train.import_meta_graph(path + '/my_model.ckpt.meta')
    saver.restor(session, path + 'my_model.ckpt.meta')
    onehot_pred = session.run('prediction:0', feed_dict=feed_dict)

# Build Comptutational Graph
def build_graph(cc):

    with tf.Graph().as_default():
        # tf Graph input
        # Placeholder definitions
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

        with tf.Session(config=config) as session: #without config for gpu support config=config
            session.run(tf.global_variables_initializer())
            train_network(session, prediction, x, y, cost, optimizer, accuracy)
            #print("Save the model")
            #save_model(session, '/home/alfu/Desktop/code_complete_model.ckpt')
            print("OKOK")
            print(cc.get_input)
            print(x)
            print(prediction)
            results = test_network(session, cc.get_input(), x, prediction)
            print(results)
            cc.setOutput(results)

if __name__ == '__main__':
    build_graph()
"""