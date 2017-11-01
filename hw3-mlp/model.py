# -*- coding: utf-8 -*-

import tensorflow as tf

epsilon = 1E-3


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.01,
                 learning_rate_decay_factor=1):
        self.x_ = tf.placeholder(tf.float32, [None, 28*28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        # TODO:  implement input -- Linear -- BN -- ReLU -- Linear -- loss
        #        the 10-class prediction output is named as "logits"
        # linear layer1
        self.w1 = weight_variable([784, 81])
        self.b1 = bias_variable([81])
        x_l1 = tf.matmul(self.x_, self.w1) + self.b1
        x_bn = batch_normalization_layer(x_l1, is_train)
        # x_bn = x_l1
        x_l1_relu = tf.nn.relu(x_bn)

        # linear layer2
        self.w2 = weight_variable([81, 10])
        self.b2 = bias_variable([10])
        logits = tf.matmul(x_l1_relu, self.w2) + self.b2

        # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.y_)
        self.pred = tf.argmax(logits, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# def batch_normalization_layer(inputs, isTrain=True, isTest=False):  # for mlp
#     # TODO: implemented the batch normalization func and applied it on fully-connected layers
#     mean_sum = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]), trainable=False)
#     var_sum = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]), trainable=False)
#     iter_num = tf.Variable(initial_value=0.0, expected_shape=[], dtype=tf.float32, trainable=False)
#     epsilon = 1E-3
#     gamma = tf.Variable(initial_value=1.0)
#     beta = tf.Variable(initial_value=0.0)
#     if isTrain:
#         mean, var = tf.nn.moments(inputs, axes=[0])
#         inputs_normalized = tf.divide(tf.subtract(inputs, mean), tf.sqrt(var + epsilon))
#         inputs_bn = tf.add(tf.multiply(inputs_normalized, gamma), beta)
#         return inputs_bn, mean_sum, var_sum, iter_num
#     elif not isTest:
#         mean, var = tf.nn.moments(inputs, axes=[0])
#         assigned_mean = tf.assign(mean_sum, tf.add(mean_sum, mean))
#         assigned_var = tf.assign(var_sum, tf.add(var_sum, tf.divide(tf.multiply(var, 100), 99))) #  !!!!!!!!!!!!!TO DO
#         train_iter = tf.assign(iter_num, tf.add(iter_num, 1.0))
#         with tf.control_dependencies([assigned_mean, assigned_var, train_iter]):
#             return inputs, mean_sum, var_sum, iter_num
#     else:
#         inputs_normalized = tf.divide(tf.subtract(inputs, tf.divide(mean_sum, iter_num)),
#                                       tf.sqrt(tf.divide(var_sum, iter_num) + epsilon))
#         inputs_bn = tf.add(tf.multiply(inputs_normalized, gamma), beta)
#         return inputs_bn, mean_sum, var_sum, iter_num


def batch_normalization_layer(inputs, isTrain=True):
    # TODO: implemented the batch normalization func and applied it on fully-connected layers
    ema_factor = 0.999
    epsilon = 1E-3

    gamma = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    train_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    train_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if isTrain:
        mean, var = tf.nn.moments(inputs, axes=[0])
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