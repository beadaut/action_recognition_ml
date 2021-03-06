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


# def inception2d(x, in_channels, filter_count, is_training, bn_decay, name='incp'):
def inception2d(x, filter_count, is_training, bn_decay, name='incp'):

    filter_count = int(filter_count/4)

    # # 1x1
    one_by_one = tf_ops.conv2d(x, filter_count, [1, 1],
                        padding='SAME', stride=[1, 1],
                        bn=True, is_training=is_training,
                               scope=name+'1by1', bn_decay=bn_decay)

    # print("\none by one shape: ", one_by_one.shape)

    # 3x3
    three_by_three = tf_ops.conv2d(one_by_one, filter_count, [3, 3],
                               padding='SAME', stride=[1, 1],
                               bn=True, is_training=is_training,
                               scope=name+'3by3', bn_decay=bn_decay)

    # print("\nthree by three shape: ", three_by_three.shape)

    # 5x5
    five_by_five = tf_ops.conv2d(one_by_one, filter_count, [5, 5],
                               padding='SAME', stride=[1, 1],
                               bn=True, is_training=is_training,
                               scope=name+'5by5', bn_decay=bn_decay)
    # print("\nfive by five shape: ", five_by_five.shape)


    # avg pooling
    pooling = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[
                             1, 1, 1, 1], padding='SAME')

    one_by_one_pool = tf_ops.conv2d(pooling, filter_count, [1, 1],
                               padding='SAME', stride=[1, 1],
                               bn=True, is_training=is_training,
                               scope=name+'1by1_pool', bn_decay=bn_decay)

    # print("\npooling shape: ", one_by_one_pool.shape)

    # Concat in the 4th dim to stack
    x = tf.concat([one_by_one, three_by_three, five_by_five, one_by_one_pool], axis=-1)
    # x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)


def build_graph_multi(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    print("\nNetowrk Input: ", input_pl)

    with tf.device('/device:GPU:1'):
        net = tf.image.resize_images(input_pl, [cfg.im_dim, cfg.im_dim])

        net = tf_ops.conv2d(net, 32, [5, 5],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.conv2d(net, 32, [5, 5],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv1_1', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.conv2d(net, 48, [5, 5],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.conv2d(net, 48, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv2_2', bn_decay=bn_decay)

        net = do_bn(net, is_training)

        net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

        net = tf_ops.conv2d(net, 64, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.conv2d(net, 64, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

    with tf.device('/device:GPU:2'):
        net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

        net = tf_ops.conv2d(net, 128, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.conv2d(net, 128, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv6', bn_decay=bn_decay)

        print("\nfinal conv shape: ", net.shape)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        # Symmetric function: max pooling
        # net = tf_ops.max_pool2d(net, [3,3], padding='VALID', scope='maxpool')

        # net = tf.reshape(net, [cfg.batch_size, -1])
        net = tf.contrib.layers.flatten(net)

        print("\nshape after flatenning: ", net.shape)

        net = tf_ops.fully_connected(net, 1024, bn=True, is_training=is_training,
                                    scope='fc1', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                            scope='dp1')

        net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='fc2', bn_decay=bn_decay)

        net = tf.nn.relu(net)

        net = do_bn(net, is_training)

        net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                            scope='dp2')

        net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

        # net = tf.nn.sigmoid(net, name="output_node")
        net = tf.nn.softmax(net, name="output_node")

        print("\nShape of logits: ", net.shape)

    return net


def build_graph(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    print("\nNetowrk Input: ", input_pl)

    net = tf.image.resize_images(input_pl, [cfg.im_dim, cfg.im_dim])

    net = tf_ops.conv2d(net, 32, [5, 5],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 48, [5, 5],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1_1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool_1')

    net = inception2d(net, 64, is_training, bn_decay, name='incp_1')

    net = do_bn(net, is_training)

    net = inception2d(net, 64, is_training, bn_decay, name='incp_2')

    net = do_bn(net, is_training)

    net = inception2d(net, 64, is_training, bn_decay, name='incp_3')

    net = do_bn(net, is_training)

    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool_2')

    net = inception2d(net, 96, is_training, bn_decay, name='incp_4')

    net = do_bn(net, is_training)

    net = inception2d(net, 96, is_training, bn_decay, name='incp_5')

    net = do_bn(net, is_training)

    net = inception2d(net, 128, is_training, bn_decay, name='incp_6')

    net = do_bn(net, is_training)

    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool_3')
    
    print("\nfinal conv shape: ", net.shape)

    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 1024, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    # net = tf.nn.sigmoid(net, name="output_node")
    net = tf.nn.softmax(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net


def build_graph_normal(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    print("\nNetowrk Input: ", input_pl)

    net = tf.image.resize_images(input_pl, [cfg.im_dim, cfg.im_dim])

    net = tf_ops.conv2d(net, 32, [5, 5],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 32, [5, 5],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv1_1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 48, [5, 5],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 48, [3, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv2_2', bn_decay=bn_decay)

    net = do_bn(net, is_training)

    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 64, [3, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv3', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 64, [3, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')

    net = tf_ops.conv2d(net, 128, [3, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.conv2d(net, 128, [3, 3],
                        padding='VALID', stride=[1, 1],
                        bn=True, is_training=is_training,
                        scope='conv6', bn_decay=bn_decay)

    print("\nfinal conv shape: ", net.shape)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [3,3], padding='VALID', scope='maxpool')

    # net = tf.reshape(net, [cfg.batch_size, -1])
    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ", net.shape)

    net = tf_ops.fully_connected(net, 1024, bn=True, is_training=is_training,
                                 scope='fc1', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp1')

    net = tf_ops.fully_connected(net, 512, bn=True, is_training=is_training,
                                 scope='fc2', bn_decay=bn_decay)

    net = tf.nn.relu(net)

    net = do_bn(net, is_training)

    net = tf_ops.dropout(net, keep_prob=keep_prob, is_training=is_training,
                         scope='dp2')

    net = tf_ops.fully_connected(net, cfg.num_classes, scope='fc3')

    # net = tf.nn.sigmoid(net, name="output_node")
    net = tf.nn.softmax(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net

def get_loss(pred, label):
    """ 
    pred: B*NUM_CLASSES,
    label: B, 
    """
    # with tf.device('/device:GPU:3'):
    label_one_hot = tf.one_hot(label, cfg.num_classes)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label_one_hot)
    loss = tf.keras.backend.categorical_crossentropy(label_one_hot, pred)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss
