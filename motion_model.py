import os
import sys
import numpy as np
import tensorflow as tf
import utils.tf_ops as tf_ops
from utils.config import cfg


def build_graph(input_pl, is_training, weight_decay=0.0, keep_prob=1.0, bn_decay=None):

    print("\nNetowrk Input: ", input_pl)

    net = tf.image.resize_images(input_pl, [cfg.im_dim, cfg.im_dim])

    net = tf_ops.conv2d(net, 32, [5, 5],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    
    net = tf.nn.relu(net)
    
    net = tf_ops.conv2d(net, 48, [5,5],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    
    net = tf.nn.relu(net)
    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')
    
    net = tf_ops.conv2d(net, 64, [3,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    
    net = tf.nn.relu(net)

    net = tf_ops.conv2d(net, 64, [3,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    
    net = tf.nn.relu(net)
    net = tf_ops.max_pool2d(net, [3, 3], padding='VALID', scope='maxpool')
    
    net = tf_ops.conv2d(net, 128, [3,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    print("\nfinal conv shape: ", net.shape)
    

    net = tf.nn.relu(net)

    # Symmetric function: max pooling
    # net = tf_ops.max_pool2d(net, [3,3], padding='VALID', scope='maxpool')

    # net = tf.reshape(net, [cfg.batch_size, -1])
    net = tf.contrib.layers.flatten(net)

    print("\nshape after flatenning: ",net.shape)

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

    # net = tf.nn.sigmoid(net, name="output_node")
    net = tf.nn.softmax(net, name="output_node")

    print("\nShape of logits: ", net.shape)

    return net

def get_loss(pred, label):
    """ 
    pred: B*NUM_CLASSES,
    label: B, 
    """
    
    label_one_hot = tf.one_hot(label, cfg.num_classes)
    # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label_one_hot)
    loss = tf.keras.backend.categorical_crossentropy(label_one_hot, pred)

    

    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss
