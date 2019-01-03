"""
one shot learning training pipeline
"""
# from __future__ import print_function
import os
import sys
import numpy as np
import math
import random

import tensorflow as tf

from tqdm import tqdm
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.tensorboard.plugins import projector

from utils.ops import *
from utils.config import cfg
from utils.data_generator_utils import DataGenerator, load_npy_filenames, jitter_point_cloud, NewTripletGenerator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_NAME = cfg.model_name

if MODEL_NAME=="simple_ff":
    from triple_loss_depth_image_model import build_graph, get_loss
else:
    print("The model name give in the config file is not available, please check and try again!")
    raise

LOGDIR = cfg.logdir+MODEL_NAME+"_"+str(cfg.num_frames)+"_"+str(cfg.im_dim)
if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)

log_screen_file = LOGDIR + '/log_screen.txt'
if os.path.exists(log_screen_file):
    os.remove(log_screen_file)

# fout = open(log_screen_file,'w')

EMBED_LOG_DIR = os.getcwd()+'/embed'
VISUALIZE_NAME = "anchor_embedding"
METADATA_PATH = EMBED_LOG_DIR+'/metadata.tsv'

BUFFER_SAMPLES = []

def get_bn_decay(step):
    bn_momentum = tf.train.exponential_decay(
                      0.5,
                      step * cfg.batch_size,
                      float(cfg.decay_step),
                      0.5,
                      staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

def F1_score(precision, recall):
    f1_score = 2*((precision*recall)/(precision+recall))
    return f1_score

def MCC(TP, TN, FP, FN):
    mcc = ((TP*TN)-(FP*FN))/tf.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    return mcc

def placeholder_inputs(batch_size, num_frames):
    # batch size should be Nnone
    inputs_pl = tf.placeholder(tf.float32, shape=(
        None, 240, 320, num_frames))
    labels_pl = tf.placeholder(tf.int32, shape=(None))
    return inputs_pl, labels_pl

def create_embedding():
    print('***** Config *****')
    print('***** Building Point {}...'.format(MODEL_NAME))
    print('** num_points: {}'.format(cfg.num_points))
    print('** num_frames: {}'.format(cfg.num_frames))
    print('** num_classes: {}'.format(cfg.num_classes))
    print('** batch_size: {}'.format(cfg.batch_size))
    print('** epoch: {}'.format(cfg.epoch))
    print('** init_learning_rate: {}'.format(cfg.init_learning_rate))
    print('** decay_step: {}'.format(cfg.decay_step))
    print('** decay_rate: {}'.format(cfg.decay_rate))
    print('** weight_decay: {}'.format(cfg.weight_decay))
    print('** feature transformation: {}'.format(cfg.feat_transform))

    with tf.Graph().as_default():
        anchor, labels = placeholder_inputs(cfg.batch_size, cfg.num_frames)
        # input_negative, labels = placeholder_inputs(
        #     cfg.batch_size, cfg.num_frames)
        # input_positive, labels = placeholder_inputs(
        #     cfg.batch_size, cfg.num_frames)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        keep_prob = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, dtype=tf.int64)
        
        bn_decay = True

        tf.summary.scalar('bn_decay', bn_decay)

        # Get model and loss
        anchor_embed = build_graph(
            anchor, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay, reuse_layers=False)

        # loss, filter_loss = get_loss(anchor_embed, input_positive_embed,
        #                               input_negative_embed)

        print("\nplaceholders loaded...")

        # %% restore a previous model
        sess = tf.InteractiveSession()
        load_model_path = LOGDIR+'/model_epoch_{}'.format(cfg.load_model_epoch)
        saver = tf.train.Saver()
        saver.restore(sess, load_model_path)

        print("\nModel restored...", load_model_path)

        # Plot Variable Histogram
        t_vars = tf.trainable_variables()
        # Count number of trainable parameters
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print('************ The Number of Trainable Parameters: {} ************'.format(num_params))
        num_g_params = np.sum([np.prod(v.get_shape().as_list())
                               for v in tf.global_variables()])
        print('************ The Number of Global Parameters: {} ************'.format(num_g_params))

        print("\nEmbedding session initialized...\n")

        # classes = [0, 1, 4, 5, 7, 8]
        # classes = [2,3,6]

        # classes = [0, 3, 8, 15, 18]
        classes = [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20]

        # load datasets:
        test_dataset = np.load(
            '/media/tjosh/vault/MSRAction3D/one_shot_test_for_known.npy')

        test_data_gen = NewTripletGenerator(
            test_dataset, classes=classes, batch_size=cfg.batch_size)

        current_data, current_label = next(test_data_gen.generator)

        embedding_var = tf.Variable(tf.zeros(
            (cfg.batch_size, anchor_embed.get_shape()[1].value)), name=VISUALIZE_NAME)
        embedding_assign = embedding_var.assign(anchor_embed)
        summary_writer = tf.summary.FileWriter(EMBED_LOG_DIR)

        # create embedding projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Specify where you find the metadata
        embedding.metadata_path = 'metadata.tsv'

        # # Specify where you find the sprite (we dont need this for our case)
        # embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
        # embedding.sprite.single_image_dim.extend([28,28])

        # Say that you want to visualise the embeddings
        projector.visualize_embeddings(summary_writer, config)

        # run session to evaluate the embedding tensor
        embedding_feed_dict = {
            anchor: current_data[:, 0, :, :],
            keep_prob: 1.0,
            is_training_pl: False
        }

        sess.run(embedding_assign, feed_dict=embedding_feed_dict)

        # save the data and checkpoint
        new_saver = tf.train.Saver()
        new_saver.save(sess, "embed/model.ckpt", 1)
        print("logdir: ", "embed/model.ckpt")

        # save the metadata
        with open(METADATA_PATH, 'w') as f:
          f.write("Index\tLabel\n")
          for index, label in enumerate(current_label):
              f.write("%d\t%d\n" % (index, label[0]))


def test_distance():

    # print('***** Config *****')
    # print('***** Building Point {}...'.format(MODEL_NAME))
    # print('** num_points: {}'.format(cfg.num_points))
    # print('** num_frames: {}'.format(cfg.num_frames))
    # print('** num_classes: {}'.format(cfg.num_classes))
    # print('** batch_size: {}'.format(cfg.batch_size))
    # print('** epoch: {}'.format(cfg.epoch))
    # print('** init_learning_rate: {}'.format(cfg.init_learning_rate))
    # print('** decay_step: {}'.format(cfg.decay_step))
    # print('** decay_rate: {}'.format(cfg.decay_rate))
    # print('** weight_decay: {}'.format(cfg.weight_decay))
    # print('** feature transformation: {}'.format(cfg.feat_transform))

    with tf.Graph().as_default():
        anchor, labels = placeholder_inputs(
            cfg.batch_size, cfg.num_frames)
        input_negative, labels = placeholder_inputs(
            cfg.batch_size, cfg.num_frames)
        input_positive, labels = placeholder_inputs(
            cfg.batch_size, cfg.num_frames)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        keep_prob = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, dtype=tf.int64)
        bn_decay = True

        # Get model and loss
        anchor_embed = build_graph(
            anchor, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay, reuse_layers=False)

        input_positive_embed = build_graph(
            input_positive, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay)

        input_negative_embed = build_graph(
            input_negative, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay)

        loss, filter_loss = get_loss(anchor_embed, input_positive_embed,
                                      input_negative_embed)

        emb_distance = tf.reduce_mean(tf.reduce_sum(
            tf.square(anchor_embed - input_negative_embed), 1))

        embed_positive = tf.reduce_sum(
            tf.square(tf.subtract(anchor_embed, input_positive_embed)), axis=-1)
        embed_negative = tf.reduce_sum(
            tf.square(tf.subtract(anchor_embed, input_negative_embed)), axis=-1)
        alpha = 2.0
        basic_loss = tf.add(tf.subtract(embed_positive, embed_negative), alpha)
        # embed_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        print("\nplaceholders loaded...")
        # raise

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(config=config)

        # load_model_path = LOGDIR+'/model_epoch_{}'.format(cfg.load_model_epoch)
        load_model_path = '/media/tjosh/vault/MSRAction3D/trained_models/logdir_one_shot_128_simple_ff_5_96/model_epoch_50'
        # load_model_path = 'logdir_one_shot_re_simple_ff_5_96/model_epoch_100'
        saver = tf.train.Saver()
        saver.restore(sess, load_model_path)
        print("\nModel restored...", load_model_path)
        print("\n\n\n")

        # # for untrained model:
        # sess.run(tf.global_variables_initializer())

        # raise

        # Plot Variable Histogram
        t_vars = tf.trainable_variables()

        # Count number of trainable parameters
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print(
            '************ The Number of Trainable Parameters: {} ************'.format(num_params))
        num_g_params = np.sum([np.prod(v.get_shape().as_list())
                               for v in tf.global_variables()])
        print(
            '************ The Number of Global Parameters: {} ************'.format(num_g_params))

        print("\nEmbedding session initialized...\n")

        # class_list = [2,4,5,6,7,9,10,11,12,13,14,16,17,19,20]
        class_list = [1, 3, 8, 15, 18]

        # load datasets:
        test_dataset = np.load(
            # '/media/tjosh/vault/MSRAction3D/one_shot_test_for_known.npy')
        '/media/tjosh/vault/MSRAction3D/one_shot_test_for_unknown.npy')
            # '/media/tjosh/vault/MSRAction3D/one_shot_train.npy')
        
        test_data_gen = NewTripletGenerator(
            test_dataset, classes=class_list, batch_size=1)

        # test_num = 1000
        test_num = 4563
        print("test number: ", test_num)

        # %% #
        # discrimination_threshold = 1.0
        # discrimination_thresholds = [discrimination_threshold]

        discrimination_thresholds = [x/100.0 for x in range(20, 200, 20)]
        print("range to test: ", discrimination_thresholds)

        for discrimination_threshold in discrimination_thresholds:
            total_discrim_distance = 0
            total_similarity_distance = 0
            correct_discrim = 0
            correct_similarity = 0
            print("\n Threshold = ", discrimination_threshold)

            total_loss = 0
            # for i in range(test_data_gen.data_size):
            for i in range(test_num):
                current_data, current_label = next(test_data_gen.generator)
                current_data = np.array(current_data)
                # print("shape of current data: ", np.shape(current_data))

                discrim_distance_feed_dict = {anchor: current_data[:, 0, :, :],
                                              input_negative: current_data[:, 1, :, :],
                                              keep_prob: 1.0,
                                              is_training_pl: False}

                similarity_distance_feed_dict = {anchor: current_data[:, 0, :, :],
                                                 input_negative: current_data[:, 2, :, :],
                                                 keep_prob: 1.0,
                                                 is_training_pl: False}

                discrim_distance = emb_distance.eval(
                    session=sess, feed_dict=discrim_distance_feed_dict)

                similarity_distance = emb_distance.eval(
                    session=sess, feed_dict=similarity_distance_feed_dict)



                total_discrim_distance += discrim_distance
                total_similarity_distance += similarity_distance

                if discrim_distance > discrimination_threshold:
                    correct_discrim += 1

                if similarity_distance < discrimination_threshold:
                    correct_similarity += 1
                
                # print("\nFor labels: ", current_label)
                # print("discrimination distance: ", discrim_distance)
                # print("similarity distance: ", similarity_distance)

                # show_sample(current_data[:, 0, :, :][0])
                # show_sample(current_data[:, 1, :, :][0])
                # show_sample(current_data[:, 2, :, :][0])

                embed_loss_feed_dict = {anchor: current_data[:, 0, :, :],
                                        input_negative: current_data[:, 1, :, :],
                                        input_positive: current_data[:, 2, :, :],
                                        keep_prob: 1.0,
                                        is_training_pl:False}
                distance_loss = basic_loss.eval(
                    session=sess, feed_dict=embed_loss_feed_dict)
                
                total_loss += distance_loss

                # print("For {}; Discrimination =  {}, Similarity = {}".format(
                #     current_label, discrim_distance, similarity_distance))

            # print("Total discrimination distance = {}, Total similarity distance = {} ". format(
            #     total_discrim_distance, total_similarity_distance))
            # print("Average discrimination distance: ",
            #       total_discrim_distance/test_num)
            # print("Average similarity distance: ",
            #       total_similarity_distance/test_num)

            # discrimination precision:
            print("True Accepts: ", correct_similarity/float(test_num))
            print("False Accepts: ", (1 - correct_discrim/float(test_num)))

            # precision = correct_discrim/float(test_num)
            # true_negative = correct_similarity/float(test_num)
            # false_negative = 1-true_negative
            # recall = precision/(precision+false_negative)
            # f1_score = 2*((precision*recall)/(precision+recall))
            # print("Recall: ", recall)
            # print("F1 Score: ", f1_score)

            # print("mean basic loss: ", total_loss/float(test_num))


def show_sample(x):
    # create plot object
    # print("shape of x: ", np.shape(x))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # fill the plot with data (auto)
    for i, (c, m) in enumerate([('r', 'o'), ('b', '^'), ('y', 'X'), ('g', 'v')]):
        xs = x[:, :, i][:, 0]
        ys = x[:, :, i][:, 1]
        zs = x[:, :, i][:, 2]
        ax.scatter(xs, ys, zs, s=1.5, c=c, marker=m)

    plt.show()


def main(_):
  print('Start Loading for Embedding ...')
  test_distance()
#   create_embedding()
  print('Finished Loading for Embedding')
  # fout.close()

if __name__ == "__main__":
    tf.app.run()
