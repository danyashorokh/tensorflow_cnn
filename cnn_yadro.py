import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from sklearn import metrics

dir_input = 'input/'

# load train data
train_patches = np.load(dir_input + 'train_patches.npy')
train_labels = np.load(dir_input + 'train_labels.npy')

# load test data
test_patches = np.load(dir_input + 'test_patches.npy')
test_labels = np.load(dir_input + 'test_labels.npy')

print(train_patches[0].shape)

tf.set_random_seed(0)  # fix random state

IMAGE_SIZE = 16  # image size
IMAGE_CHANNELS = 6  # image channels number

NUM_CLASSES = 2  # output classes number
BATCH_SIZE = 500  # batch size

x = tf.placeholder('float', shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
y = tf.placeholder('int64', shape=[None])


def cnn(x):

    input_layer = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])

    # Convolutional layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height ("same").
    # Input tensor shape: [batch_size, 16, 16, 6]
    # Output tensor shape: [batch_size, 16, 16, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input tensor shape: [batch_size, 16, 16, 32]
    # Output tensor shape: [batch_size, 8, 8, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input tensor shape: [batch_size, 8, 8, 32]
    # Output tensor shape: [batch_size, 8, 8, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input tensor shape: [batch_size, 8, 8, 64]
    # Output tensor shape: [batch_size, 4, 4, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input tensor shape: [batch_size, 4, 4, 64]
    # Output tensor shape: [batch_size, 4 * 4 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])

    # Dense layer
    # Densely connected layer with 1024 neurons
    # Input tensor shape: [batch_size, 4 * 4 * 64]
    # Output tensor shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits layer
    # Input tensor shape: [batch_size, 1024]
    # Output tensor shape: [batch_size, 2]
    output = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    return output


def train_cnn(x):

    prediction = cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                                     labels=tf.one_hot(y, NUM_CLASSES)))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(len(train_patches) // BATCH_SIZE):
                feed_dict = {x: train_patches[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                             y: train_labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]}

                _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
                epoch_loss += c

            print('%s Epoch %s completed out of %s loss: %s' % (datetime.now(), epoch + 1, hm_epochs, epoch_loss))

        argmax_prediction = tf.argmax(prediction, 1)
        argmax_y = y

        correct = tf.equal(argmax_prediction, argmax_y)

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_patches, y: test_labels}))

        # TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        # TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        # FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        # FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
        #
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f11 = 2 * precision * recall / (precision + recall)
        #
        # # v.2
        # _, recall = tf.metrics.recall(argmax_y, argmax_prediction)
        # _, precision = tf.metrics.precision(argmax_y, argmax_prediction)
        #
        # f12 = 2 * precision * recall / (precision + recall)

t_start = datetime.now()
print(t_start)

train_cnn(x)

t_finish = datetime.now()

print('Training time = %s min' % str((t_finish - t_start)/timedelta(minutes=1)))
