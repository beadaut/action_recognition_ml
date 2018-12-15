import numpy as np
import tensorflow as tf

def batch_norm(input, 
              is_training):
    return tf.layers.batch_normalization(input, training=is_training)

def dropout(input,
            drop_rate,
            is_training):
    return tf.layers.dropout(input, rate=drop_rate, training=is_training)

def conv2d(input, 
           output_dims, 
           kernel_size=[1,1], 
           strides=[1,1],
           padding='valid', 
           activation=tf.nn.relu,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           kernel_regularizer=tf.nn.l2_loss,
           name=None,
           reuse=None):
    return tf.layers.conv2d(input, filters=output_dims, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation=activation,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer, name=name)

def dilated_conv2d(input,
                   output_dims,
                   kernel_size,
                   padding='valid',
                   dilation_rate=[1,1],
                   activation=tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   kernel_regularizer=tf.nn.l2_loss,
                   name=None,
                   reuse=None):
    return tf.layers.conv2d(input, filters=output_dims, kernel_size=kernel_size,
                     padding=padding, dilation_rate=dilation_rate, 
                     activation=activation, kernel_initializer=kernel_initializer, 
                     kernel_regularizer=kernel_regularizer, name=name)

def maxPooling2d(input,
                pool_size,
                strides=[1,1],
                padding='VALID',
                name=None):
    poolH, poolW = pool_size
    strideH, strideW = strides
    return tf.nn.max_pool(input, ksize=[1,poolH,poolW,1],
                          strides=[1,strideH,strideW,1],
                          padding=padding, name=name)
                          
def avgPooling2d(input,
                 pool_size,
                 strides=[1,1],
                 padding='valid',
                 name=None):
    return tf.layers.average_pooling2d(input, pool_size=pool_size,
                                       strides=strides, padding=padding,
                                       name=name)

def fully_connected(input,
                    output_dims,
                    activation=tf.nn.relu,
                    name=None):
    shape = [input.get_shape()[-1].value, output_dims] 
    W = get_weight_var(name+'/fc_weight', shape=shape)
    b = get_bias_var(name+'/fc_bias', shape=[output_dims])

    output = tf.matmul(input, W) + b
    if activation is not None:
        output = activation(output)
    return output

def get_weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())

def get_bias_var(name, shape):
	return tf.get_variable(name=name, shape=shape,
                            initializer=tf.zeros_initializer)
