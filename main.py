import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

dir_input = 'input/'

train_patches = np.load(dir_input + 'train_patches.npy')
train_labels = np.load(dir_input + 'train_labels.npy')

test_patches = np.load(dir_input + 'test_patches.npy')
test_labels = np.load(dir_input + 'test_labels.npy')

print(train_patches)
print(train_labels)


print(len(train_patches))
print(len(test_patches))


n_classes = 2
batch_size = 100

x = tf.placeholder('float', [None, 16*16])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 6], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'w_conv1': tf.Variable(tf.random_normal([5, 5, 6, 32])),  # 5, 5, 1, 32
               'w_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'w_fc': tf.Variable(tf.random_normal([4*4*64, 1024])),  # 7*7
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 16, 16, 6])

    conv1 = conv2d(x, weights['w_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['w_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 4*4*64])  # 7*7
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['w_fc']), biases['b_fc']))

    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.add(tf.matmul(fc, weights['out']), biases['out'])


    return output


def train_neural_network(x):

    prediction = convolutional_neural_network(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(len(train_patches) // batch_size):
                feed_dict = {x: train_patches[i * batch_size:(i + 1) * batch_size],
                             y: train_labels[i * batch_size:(i + 1) * batch_size]}

                _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_patches, y: test_labels}))

train_neural_network(x)
