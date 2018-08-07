import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta
from sklearn import metrics

dir_input = 'input/'

train_patches = np.load(dir_input + 'train_patches.npy')
train_labels = np.load(dir_input + 'train_labels.npy')

test_patches = np.load(dir_input + 'test_patches.npy')
test_labels = np.load(dir_input + 'test_labels.npy')

# test
t_size = 5000
train_patches = np.concatenate((train_patches[:t_size], train_patches[-t_size:]))
train_labels = np.concatenate((train_labels[:t_size], train_labels[-t_size:]))
test_patches = np.concatenate((test_patches[:t_size], test_patches[-t_size:]))
test_labels = np.concatenate((test_labels[:t_size], test_labels[-t_size:]))

np.save(dir_input + 'small_train_patches.npy', train_patches)
np.save(dir_input + 'small_train_labels.npy', train_labels)
np.save(dir_input + 'small_test_patches.npy', test_patches)
np.save(dir_input + 'small_test_labels.npy', test_labels)

exit()

print(len(train_patches))
print(train_patches[0].shape)

tf.set_random_seed(0)

IMAGE_SIZE = 16
IMAGE_CHANNELS = 6

NUM_CLASSES = 2
BATCH_SIZE = 500

x = tf.placeholder('float', shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
y = tf.placeholder('int64', shape=[None])


def cnn(x):

    input_layer = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 16, 16, 32]
    # Output Tensor Shape: [batch_size, 8, 8, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 8, 8, 32]
    # Output Tensor Shape: [batch_size, 8, 8, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 8, 8, 64]
    # Output Tensor Shape: [batch_size, 4, 4, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 4, 4, 64]
    # Output Tensor Shape: [batch_size, 4 * 4 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 4 * 4 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    output = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    return output


def train_cnn(x):

    prediction = cnn(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=tf.one_hot(y, NUM_CLASSES)))

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


# 2018-05-29 10:40:47.019255
# 2018-05-29 10:46:44.811883 Epoch 1 completed out of 10 loss: 5418.902016327147
# 2018-05-29 10:52:37.535540 Epoch 2 completed out of 10 loss: 544.2676258957363
# 2018-05-29 11:32:14.045132 Epoch 3 completed out of 10 loss: 292.21028113976354
# 2018-05-29 11:42:11.045421 Epoch 4 completed out of 10 loss: 264.3411089053261
# 2018-05-29 11:52:28.410208 Epoch 5 completed out of 10 loss: 228.79144700716643
# 2018-05-29 11:58:20.893716 Epoch 6 completed out of 10 loss: 347.66179316881124
# 2018-05-29 12:04:42.367687 Epoch 7 completed out of 10 loss: 544.5814690636653
# 2018-05-29 12:10:33.006995 Epoch 8 completed out of 10 loss: 289.0318987420178
# 2018-05-29 12:50:16.223302 Epoch 9 completed out of 10 loss: 257.2835987842991
# 2018-05-29 13:04:28.912487 Epoch 10 completed out of 10 loss: 267.65256672833493

# Process finished with exit code 137 (interrupted by signal 9: SIGKILL)

# 2018-05-29 13:19:02.822745 Epoch 1 completed out of 10 loss: 6414.275468234842
# 2018-05-29 13:24:58.770487 Epoch 2 completed out of 10 loss: 416.25010985624976
# 2018-05-29 13:30:48.582920 Epoch 3 completed out of 10 loss: 226.82338501722552
# 2018-05-29 13:49:37.294407 Epoch 4 completed out of 10 loss: 228.31862263567746
# 2018-05-29 13:55:27.590219 Epoch 5 completed out of 10 loss: 151.67859579938158
# 2018-05-29 14:12:07.707235 Epoch 6 completed out of 10 loss: 176.13612923119217
# 2018-05-29 14:17:59.580634 Epoch 7 completed out of 10 loss: 174.3668554674132
# 2018-05-29 14:32:37.689864 Epoch 8 completed out of 10 loss: 200.4157704377867
# 2018-05-29 14:38:31.467441 Epoch 9 completed out of 10 loss: 169.84641691233583
# 2018-05-29 15:08:00.137249 Epoch 10 completed out of 10 loss: 4929.881377357835
