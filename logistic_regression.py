import pandas as pd
import tensorflow as tf
import numpy as np
#import sklearn.metrics as skm
import matplotlib.pyplot as plt

data_file = 'data_final.csv'
label_file = 'labels_final.csv'
training_set_size = 0.9

print('Assembling training data...')
data = pd.read_csv(data_file, header=None)
labels = pd.read_csv(label_file, header=None)

data = data.drop([0], axis=1)
labels = labels.drop([0], axis=1)

data = data.apply(lambda x: (x - min(x))/(max(x) - min(x)))  

total_examples = data.shape[0]

training_set_split_loc = int(total_examples*training_set_size)

training_x = data[:training_set_split_loc].as_matrix()
test_x = data[training_set_split_loc:].as_matrix()

training_y = labels[:training_set_split_loc].as_matrix()
test_y = labels[training_set_split_loc:].as_matrix()

# Parameters
learning_rate = 0.001
training_iters = training_x.shape[0]
batch_size = 100
display_step = 1

# Network Parameters
n_input = 624 
n_classes = 1 

# Create model
x = tf.placeholder(tf.float32, [None, n_input])
#W = tf.Variable(tf.zeros([n_input, n_classes]))
W = tf.Variable(tf.random_normal([n_input, n_classes], stddev=0.001))
b = tf.Variable(tf.truncated_normal([n_classes]))
y_conv = tf.nn.sigmoid(tf.matmul(x, W) + b)
y = tf.placeholder(tf.float32, [None, n_classes])

#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#reg_constant = 10  # Choose an appropriate one.

regularizer = tf.nn.l2_loss(W)
reg_constant = 100000

loss = -(y * tf.log(y_conv + 1e-12) + (1 - y) * tf.log( 1 - y_conv + 1e-12))
cross_entropy = tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=[1]) + reg_constant * regularizer)

# Define loss and optimizer 
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Evaluate model
correct_pred = tf.equal(tf.to_float(tf.greater(y_conv, 0.5)), y)
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
            loss = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y})
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