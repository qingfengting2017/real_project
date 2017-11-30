from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class VGGNet:
    def __init__(self, type_number, image_size, image_channel, batch_size):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size
        pass
    def vgg_16(self, input_op, **kw):
        first_out = 32

        conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)

        conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out * 2, stripe_height=1, stripe_width=1)
        pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)

        conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out * 4, stripe_height=1, stripe_width=1)
        pool_3 = self._max_pool_op(conv_3_3, "pool_3", 2, 2, stripe_height=2, stripe_width=2)

        conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_2 = self._conv_op(conv_4_1, "conv_4_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_4 = self._max_pool_op(conv_4_3, "pool_4", 2, 2, stripe_height=2, stripe_width=2)

        conv_5_1 = self._conv_op(pool_4, "conv_5_1", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_2 = self._conv_op(conv_5_1, "conv_5_2", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        conv_5_3 = self._conv_op(conv_5_2, "conv_5_3", 3, 3, n_out=first_out * 8, stripe_height=1, stripe_width=1)
        pool_5 = self._max_pool_op(conv_5_3, "pool_5", 2, 2, stripe_height=2, stripe_width=2)

        shp = pool_5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        reshape_pool_5 = tf.reshape(pool_5, [-1, flattened_shape], name="reshape_pool_5")

        fc_6 = self._fc_op(reshape_pool_5, name="fc_6", n_out=2048)
        fc_6_drop = tf.nn.dropout(fc_6, keep_prob=kw["keep_prob"], name="fc_6_drop")

        fc_7 = self._fc_op(fc_6_drop, name="fc_7", n_out=1024)
        fc_7_drop = tf.nn.dropout(fc_7, keep_prob=kw["keep_prob"], name="fc_7_drop")

        fc_8 = self._fc_op(fc_6_drop, name="fc_8", n_out=self._type_number)
        softmax = tf.nn.softmax(fc_8)
        prediction = tf.argmax(softmax, 1)

        return fc_8, softmax, prediction
        # conv_1_1 = self._conv_op(input_op, "conv_1_1", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        # conv_1_2 = self._conv_op(conv_1_1, "conv_1_2", 3, 3, n_out=first_out, stripe_height=1, stripe_width=1)
        # pool_1 = self._max_pool_op(conv_1_2, "pool_1", 2, 2, stripe_height=2, stripe_width=2)
        #
        # conv_2_1 = self._conv_op(pool_1, "conv_2_1", 3, 3, n_out=first_out*2, stripe_height=1, stripe_width=1)
        # conv_2_2 = self._conv_op(conv_2_1, "conv_2_2", 3, 3, n_out=first_out*2, stripe_height=1, stripe_width=1)
        # pool_2 = self._max_pool_op(conv_2_2, "pool_2", 2, 2, stripe_height=2, stripe_width=2)
        #
        # conv_3_1 = self._conv_op(pool_2, "conv_3_1", 3, 3, n_out=first_out*4, stripe_height=1, stripe_width=1)
        # conv_3_2 = self._conv_op(conv_3_1, "conv_3_2", 3, 3, n_out=first_out*4, stripe_height=1, stripe_width=1)
        # conv_3_3 = self._conv_op(conv_3_2, "conv_3_3", 3, 3, n_out=first_out*4, stripe_height=1, stripe_width=1)
        # pool_3 = self._max_pool_op(conv_3_3, "pool_3", 2, 2, stripe_height=2, stripe_width=2)
        #
        # conv_4_1 = self._conv_op(pool_3, "conv_4_1", 3, 3, n_out=first_out*8, stripe_height=1, stripe_width=1)
        # conv_4_2 = self._conv_op(conv_4_1, "conv_4_1", 3, 3, n_out=first_out*8, stripe_height=1, stripe_width=1)
        # conv_4_3 = self._conv_op(conv_4_2, "conv_4_3", 3, 3, n_out=first_out*8, stripe_height=1, stripe_width=1)
        # pool_4 = self._max_pool_op(conv_4_3, "pool_4", 2, 2, stripe_height=2, stripe_width=2)
        #
        # conv_5_1 = self._conv_op(pool_4, "conv_5_1", 3, 3, n_out=first_out*16, stripe_height=1, stripe_width=1)
        # conv_5_2 = self._conv_op(conv_5_1, "conv_5_2", 3, 3, n_out=first_out*16, stripe_height=1, stripe_width=1)
        # conv_5_3 = self._conv_op(conv_5_2, "conv_5_3", 3, 3, n_out=first_out*16, stripe_height=1, stripe_width=1)
        # pool_5 = self._max_pool_op(conv_5_3, "pool_5", 2, 2, stripe_height=2, stripe_width=2)
        #
        # shp = pool_5.get_shape()
        # flattened_shape = shp[1].value * shp[2].value * shp[3].value
        # reshape_pool_5 = tf.reshape(pool_5, [-1, flattened_shape], name="reshape_pool_5")
        #
        # fc_6 = self._fc_op(reshape_pool_5, name="fc_6", n_out=2048)
        # fc_6_drop = tf.nn.dropout(fc_6, keep_prob=kw["keep_prob"], name="fc_6_drop")
        #
        # fc_7 = self._fc_op(fc_6_drop, name="fc_7", n_out=1024)
        # fc_7_drop = tf.nn.dropout(fc_7, keep_prob=kw["keep_prob"], name="fc_7_drop")
        #
        # fc_8 = self._fc_op(fc_7_drop, name="fc_8", n_out=self._type_number)
        # softmax = tf.nn.softmax(fc_8)
        # prediction = tf.argmax(softmax, 1)
        # return fc_8, softmax, prediction
    # 创建卷积层
    @staticmethod
    def _conv_op(input_op, name, kernel_height, kernel_width, n_out, stripe_height, stripe_width):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name=name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, filter=kernel, strides=(1, stripe_height, stripe_width, 1), padding="SAME")
            biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
            activation = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
            # tf.histogram_summary(name, kernel)
            return activation
        pass

    # 创建全连接层
    @staticmethod
    def _fc_op(input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value
        with tf.name_scope(name) as scope:
            kernel = tf.get_variable(scope + "w", shape=[n_in, n_out], dtype=tf.float32,
                                     initializer=tcl.xavier_initializer())
            biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b")
            activation = tf.nn.relu_layer(x=input_op, weights=kernel, biases=biases, name=scope)
            return activation
        pass

    # 最大池化层
    @staticmethod
    def _max_pool_op(input_op, name, kernel_height, kernel_width, stripe_height, stripe_width):
        return tf.nn.max_pool(input_op, ksize=[1, kernel_height, kernel_width, 1],
                              strides=[1, stripe_height, stripe_width, 1], padding="SAME", name=name)

    pass


# Initialize the variables (i.e. assign their default value)
keep_prob={}
keep_prob["keep_prob"]=0.7


vgg_16 = VGGNet(10, 28, 1, 100)
images = tf.placeholder(dtype=tf.float32, shape=[100, 28, 28, 1])
batch_ys_new = tf.placeholder(dtype=tf.int64, shape=100)
logits, softmax, prediction = vgg_16.vgg_16(images, **keep_prob)
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_ys_new, logits=logits)
loss = tf.reduce_mean(entropy)
tf.summary.scalar('loss1', loss)

# global_step=25
# learning_rate = tf.train.exponential_decay(0.1, global_step=global_step, decay_steps=10, decay_rate=0.9)
# solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss)
solver = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(loss)
saver = tf.train.Saver()
merged = tf.summary.merge_all()

pgraph = "/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/tensorboard/graph"
ploss = "/root/Desktop/Dataset2/test_tensorflow/tensorflow-classification-network/src/tensorboard/loss"
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

    writer = tf.summary.FileWriter(pgraph, sess.graph)
    lwriter = tf.summary.FileWriter(ploss)

    tf.logging.set_verbosity(tf.logging.INFO)
    sess.run(tf.global_variables_initializer())
    # sess.run(learning_rate)
    print("Begin")
    # Training cycle
    for epoch in range(25):
        # print("epoch = ", epoch)
        total_batch = int(mnist.train.num_examples / 100)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_ys_1 = np.argmax(batch_ys, axis=1)
            batchxs2run = tf.constant(batch_xs, shape=[100, 28, 28, 1])
            batchys2run = tf.constant(batch_ys_1, shape=[100])
            xs, ys = sess.run([batchxs2run, batchys2run])

            summary, loss1, _ = sess.run(fetches=[merged, loss, solver], feed_dict={images: xs, batch_ys_new: ys})#fetches=[loss, solver, softmax]

            lwriter.add_summary(summary, i*epoch)
            # sess.run(merged)
            print('substep ', i, 'loss = ', loss1)
        print('epoch ', epoch, 'loss = ', loss1)

    #test
    batch_test_xs, batch_test_ys = mnist.test.next_batch(100)
    batch_ys_1 = np.argmax(batch_test_ys, axis=1)
    batchxs2run = tf.constant(batch_test_xs, shape=[100, 28, 28, 1])
    batchys2run = tf.constant(batch_ys_1, shape=[100])
    xs, ys = sess.run([batchxs2run, batchys2run])

    correct_prediction = tf.equal(prediction, batch_ys_new)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({images: xs, batch_ys_new: ys}))
    saver.save(sess, save_path="./model/vgg_16.ckpt")