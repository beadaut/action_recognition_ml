import os
import sys
import numpy as np
import tensorflow as tf
import utils.tf_ops as tf_ops
from utils.pc_config import cfg


def build_graph_old(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    print("\nNetowrk Input: ", input_pl)

    pool_num = int(cfg.num_points/3)

    net = tf_ops.conv2d(input_pl, 64, [1, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    print("\nshape after input: ", net.shape)

    net = tf.nn.relu(net)

    # skip_pool = tf_ops.max_pool2d(net, [pool_num, 1],
    #                               padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 1024, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    print("\nshape before max pool: ", net.shape)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [pool_num, 1],
    net = tf_ops.max_pool2d(net, [cfg.num_points,1],
                            padding='VALID', scope='maxpool')

    # skip_multiply = tf.multiply(net, skip_pool)

    # net = tf.add(net, skip_multiply)

    print("\nshape after skip pool: ", net.shape)

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 128, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    net = tf.nn.relu(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net
  

def build_graph_ext(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    # end_points = {}
    print("\n The skip pool model....")
    print("Netowrk Input: ", input_pl)

    pool_num = int(cfg.num_points/3)

    # to normalize the data before inputing
    # input_pl = tf.layers.batch_normalization(input_pl)

    net = tf_ops.conv2d(input_pl, 128, [1, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    print("\nshape after input: ", net.shape)

    net = tf.nn.relu(net)

    skip_pool = tf_ops.max_pool2d(net, [pool_num, 1],
                                  padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    # net = tf_ops.conv2d(net, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv3', bn_decay=bn_decay)

    # net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    print("\nshape before max pool: ", net.shape)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [cfg.num_points,1],
    net = tf_ops.max_pool2d(net, [pool_num, 1],
                            padding='VALID', scope='maxpool')

    skip_multiply = tf.multiply(net, skip_pool)

    net = tf.add(net, skip_multiply)

    print("\nshape after skip pool: ", net.shape)

    net = tf_ops.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.max_pool2d(net, [pool_num, 1],
                            padding='VALID', scope='maxpool')

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 100, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    net = tf.nn.sigmoid(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net


def build_graph(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    # end_points = {}
    print("\n The skip pool model....")
    print("Netowrk Input: ", input_pl)

    pool_num = int(cfg.num_points/3)

    net = tf_ops.conv2d(input_pl, 128, [1, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    print("\nshape after input: ", net.shape)

    net = tf.nn.relu(net)

    skip_pool = tf_ops.max_pool2d(net, [pool_num, 1],
                                  padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    print("\nshape before max pool: ", net.shape)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [cfg.num_points,1],
    net = tf_ops.max_pool2d(net, [pool_num, 1],
                            padding='VALID', scope='maxpool')

    skip_multiply = tf.multiply(net, skip_pool)

    net = tf.add(net, skip_multiply)

    print("\nshape after skip pool: ", net.shape)

    net = tf_ops.conv2d(net, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv3', bn_decay=bn_decay)
    

    net = tf.nn.relu(net)

    # net = tf_ops.max_pool2d(net, [pool_num, 1],
    #                         padding='VALID', scope='maxpool')

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 100, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    net = tf.nn.sigmoid(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net


def build_graph_stable(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    # end_points = {}
    print("\n The skip pool model....")
    print("Netowrk Input: ", input_pl)

    pool_num = int(cfg.num_points/3)

    net = tf_ops.conv2d(input_pl, 128, [1, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    print("\nshape after input: ", net.shape)

    net = tf.nn.relu(net)

    skip_pool = tf_ops.max_pool2d(net, [pool_num, 1],
                                  padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    # net = tf_ops.conv2d(net, 64, [1,1],
    #                      padding='VALID', stride=[1,1],
    #                      bn=True, is_training=is_training,
    #                      scope='conv3', bn_decay=bn_decay)

    # net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 64, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 128, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    print("\nshape before max pool: ", net.shape)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [cfg.num_points,1],
    net = tf_ops.max_pool2d(net, [pool_num, 1],
                            padding='VALID', scope='maxpool')

    skip_multiply = tf.multiply(net, skip_pool)

    net = tf.add(net, skip_multiply)

    print("\nshape after skip pool: ", net.shape)

    net = tf_ops.conv2d(net, 256, [1, 1],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv3', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    # net = tf_ops.max_pool2d(net, [pool_num, 1],
    #                         padding='VALID', scope='maxpool')

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 100, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    net = tf.nn.sigmoid(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net

def get_loss(pred, label):
    """ 
    pred: B*NUM_CLASSES,
    label: B, 
    """

    label_one_hot = tf.one_hot(label, cfg.num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred, labels=label_one_hot)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss
