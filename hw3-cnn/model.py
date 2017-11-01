# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 1, 28, 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        x = tf.reshape(self.x_, [-1, 28, 28, 1])

        # TODO: implement input -- Conv -- BN -- ReLU -- MaxPool -- Conv -- BN -- ReLU -- MaxPool -- Linear -- loss
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], padding="same")
        conv1_bn = batch_normalization_layer(conv1, is_train)
        conv1_bn_relu = tf.nn.relu(conv1_bn)
        pool1 = tf.layers.max_pooling2d(inputs=conv1_bn_relu, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same")
        conv2_bn = batch_normalization_layer(conv2, is_train)
        conv2_bn_relu = tf.nn.relu(conv2_bn)
        pool2 = tf.layers.max_pooling2d(inputs=conv2_bn_relu, pool_size=[2, 2], strides=2)

        linear = tf.reshape(pool2, [-1, 3136])
        w = weight_variable([3136, 10])
        logits = tf.matmul(linear, w)
        #        the 10-class prediction output is named as "logits"
        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                                         dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on conv and fully-connected layers
    # hint: you can add extra parameters (e.g., shape) if necessary
    ema_factor = 0.999
    epsilon = 1E-3

    gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    train_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    train_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if isTrain:
        mean, var = tf.nn.moments(inputs, axes=[0, 1, 2])
        mean_assigned = tf.assign(train_mean, train_mean * ema_factor + mean * (1 - ema_factor))
        var_assigned = tf.assign(train_var, train_var * ema_factor + var * (1 - ema_factor))
        with tf.control_dependencies([mean_assigned, var_assigned]):
            inputs_normalized = tf.divide(inputs - mean, tf.sqrt(var + epsilon))
            inputs_bn = tf.multiply(inputs_normalized, gamma) + beta
            return inputs_bn
    else:
        inputs_normalized = tf.divide(inputs - train_mean, tf.sqrt(train_var + epsilon))
        inputs_bn = tf.multiply(inputs_normalized, gamma) + beta
        return inputs_bn
