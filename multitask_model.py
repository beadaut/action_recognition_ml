"""
triple_loss_depth_image_model
"""
import os
import sys
import numpy as np
import tensorflow as tf
import utils.tf_ops as tf_ops
from utils.config import cfg



def do_bn(net, is_training):
  net = tf.layers.batch_normalization(
        net,
        training=is_training
      )
  
  return net

def build_graph(input_pl, is_training, keep_prob, weight_decay=0.0, bn_decay=None, reuse_layers=True):

    print("\nNetowrk Input: ", input_pl)

    net = tf.image.resize_images(input_pl, [cfg.im_dim, cfg.im_dim])

    net = tf.layers.conv2d(
            inputs=net,
            filters=32,
            kernel_size=[5,5],
            # strides=(1, 1),
            # padding='valid',
            # data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            # kernel_regularizer=l2_reg,
            name='conv2d_layer_1',
            reuse=reuse_layers)
          
    # implement batch normalization
    if bn_decay:
      net = do_bn(net, is_training)

    net = tf.layers.conv2d(
            inputs=net,
            filters=48,
            kernel_size=[5,5],
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='conv2d_layer_2',
            reuse=reuse_layers)
    
    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

    if bn_decay:
      net = do_bn(net, is_training)

    net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[3,3],
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='conv2d_layer_3',
            reuse=reuse_layers)
    

    if bn_decay:
      net = do_bn(net, is_training)
    

    net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[3,3],
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='conv2d_layer_4',
            reuse=reuse_layers)
    
    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

    if bn_decay:
      net = do_bn(net, is_training)
    
    
    net = tf.layers.conv2d(
            inputs=net,
            filters=128,
            kernel_size=[3,3],
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name='conv2d_layer_5',
            reuse=reuse_layers)
    

    print("\nFinal Conv Shape: ",net.shape)
    

    if bn_decay:
      net = do_bn(net, is_training)

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ",net.shape)

    net = tf.layers.dense(
      inputs=net,
      units=512,
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name='fc_1',
      reuse=reuse_layers)
    

    if bn_decay:
      net = do_bn(net, is_training)
    

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                          scope='dp1')


    net = tf.layers.dense(
      inputs=net,
      units=128,
      # units=256,
      activation=tf.nn.relu,
      use_bias=True,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      bias_initializer=tf.zeros_initializer(),
      name='fc_2',
      reuse=reuse_layers)
    
    embed_logits = tf.nn.l2_normalize(net, axis=-1)
    
    class_logits = tf.layers.dense(
        inputs=net,
        units=cfg.num_classes,
        activation=tf.nn.softmax,
        # activation=tf.nn.sigmoid,
        use_bias=True,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.zeros_initializer(),
        name='classification_output',
        reuse=reuse_layers)

    tf.summary.histogram('embed_outputs', embed_logits)

    print("\nShape of logits: ", embed_logits.shape)

    return embed_logits, class_logits


def get_loss(pred, label, anchor_embed, input_positive_embed, input_negative_embed):
    """ pred: B*NUM_CLASSES,
        label: B, """
    
    # classification:
    label_one_hot = tf.one_hot(label, cfg.num_classes)
    loss = tf.keras.backend.categorical_crossentropy(
        label_one_hot,
        pred)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label_one_hot)
    classification_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classification loss', classification_loss)

    # embedding:
    embed_positive = tf.reduce_sum(tf.square(tf.subtract(anchor_embed, input_positive_embed)),axis=-1)
    embed_negative = tf.reduce_sum(tf.square(tf.subtract(anchor_embed, input_negative_embed)),axis=-1)
    alpha = 2.0
    basic_loss = tf.add(tf.subtract(embed_positive, embed_negative), alpha)
    filter_loss = tf.reduce_mean(tf.subtract(embed_positive, embed_negative))
    embed_loss = tf.reduce_mean(tf.maximum(basic_loss,0.0),0)
    tf.summary.scalar('embedding loss', embed_loss)
    tf.summary.scalar('basic loss', tf.reduce_mean(basic_loss))

    return classification_loss, embed_loss, filter_loss, tf.reduce_mean(basic_loss)
