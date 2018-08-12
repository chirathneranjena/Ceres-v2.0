import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from nltk.metrics.association import TOTAL

data_file = 'data_final.csv'
label_file = 'labels_final.csv'
training_set_size = 0.95

print('Assembling training data...')
data = pd.read_csv(data_file, header=None)
labels = pd.read_csv(label_file, header=None)

data = data.drop([0], axis=1)
labels = labels.drop([0], axis=1)

#data = data.apply(lambda x: (x - min(x))/(max(x) - min(x)))

total_examples = data.shape[0]
print(total_examples)
training_set_split_loc = int(total_examples*training_set_size)

training_x = data[:training_set_split_loc].as_matrix()
test_x = data[training_set_split_loc:].as_matrix()

y_hot = np.empty((0, 2))

count = 0
while count < total_examples:
    if (labels.loc[count, 1] > 0):
        mini_array = [[1, 0]]
        y_hot = np.vstack((y_hot, mini_array))
    else:
        mini_array = [[0, 1]]
        y_hot = np.vstack((y_hot, mini_array))
    count += 1
        
training_y = y_hot[:training_set_split_loc]
test_y = y_hot[training_set_split_loc:]

print('Starting to train...')


# Parameters
learning_rate = 0.001
training_iters = training_x.shape[0]
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of features
n_hidden_2 = 1024 # 2nd layer number of features
n_hidden_3 = 1024 # 3rd layer number of features
n_hidden_4 = 1024 # 4th layer number of features
n_hidden_5 = 1024 # 5th layer number of features
n_hidden_6 = 1024 # 6th layer number of features
n_hidden_7 = 1024 # 7th layer number of features
n_hidden_8 = 1024 # 8th layer number of features
n_hidden_9 = 1024 # 9th layer number of features
n_hidden_10 = 1024 # 10th layer number of features
n_input = 624 
n_classes = 2


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3) 
    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)  
    # Hidden layer with RELU activation
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)   
    # Hidden layer with RELU activation
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_6 = tf.nn.relu(layer_6)   
    # Hidden layer with RELU activation
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_7 = tf.nn.relu(layer_7)
    # Hidden layer with RELU activation
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    layer_8 = tf.nn.relu(layer_8)   
    # Hidden layer with RELU activation
    layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
    layer_9 = tf.nn.relu(layer_9)   
    # Hidden layer with RELU activation
    layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
    layer_10 = tf.nn.relu(layer_10)                                    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),    
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),    
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),    
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),    
    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),    
    'h9': tf.Variable(tf.random_normal([n_hidden_8, n_hidden_9])),    
    'h10': tf.Variable(tf.random_normal([n_hidden_9, n_hidden_10])),                            
    'out': tf.Variable(tf.random_normal([n_hidden_10, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])), 
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])), 
    'b6': tf.Variable(tf.random_normal([n_hidden_6])), 
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'b8': tf.Variable(tf.random_normal([n_hidden_8])), 
    'b9': tf.Variable(tf.random_normal([n_hidden_9])), 
    'b10': tf.Variable(tf.random_normal([n_hidden_10])),                              
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy =  tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
cost_history = np.empty(shape=[1],dtype=float)
acc_history = np.empty(shape=[1],dtype=float)
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x = training_x[((step-1)*batch_size) : (step*batch_size)]
        batch_y = training_y[((step-1)*batch_size) : (step*batch_size)]

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            cost_history = np.append(cost_history, loss)
            acc_history = np.append(acc_history, acc)            
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
    #print(sess.run(tf.to_float(tf.greater(y_conv, 0.5)), feed_dict={x: test_x}))

    fig = plt.figure(figsize=(10, 8))
    plt.plot(cost_history)
    plt.axis([0, step, np.min(cost_history), np.max(cost_history)])
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    plt.plot(acc_history)
    plt.axis([0, step, np.min(acc_history), np.max(acc_history)])
    plt.xlabel('# Iterations')
    plt.ylabel('Accuracy')
    plt.show()

    #metrics
    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:test_y})

    print("validation accuracy:", val_accuracy)
    y_true = np.argmax(test_y,1)
    print("Precision", skm.precision_score(y_true, y_pred))
    print("Recall", skm.recall_score(y_true, y_pred))
    print("f1_score", skm.f1_score(y_true, y_pred))
