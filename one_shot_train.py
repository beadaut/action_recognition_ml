"""
one shot learning training pipeline
"""
import os
import sys
import numpy as np
import math
import random

import tensorflow as tf

from tqdm import tqdm
from tensorflow.python import debug as tf_debug
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

fout = open(log_screen_file,'w')

BUFFER_SAMPLES = []

def log_string(out_str):
    fout.write(out_str+'\n')
    fout.flush()
    print(out_str)

def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(
                        cfg.init_learning_rate,  # Base learning rate.
                        step * cfg.batch_size,  # Current index into the dataset.
                        cfg.decay_step,          # Decay step.
                        cfg.decay_rate,          # Minimum learning rate * init_lr.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-6) # CLIP THE LEARNING RATE!
    return learning_rate        

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

def train():
    log_string('***** Config *****')
    log_string('***** Building Point {}...'.format(MODEL_NAME))
    log_string('** im_dim: {}'.format(cfg.im_dim))
    log_string('** num_frames: {}'.format(cfg.num_frames))
    log_string('** num_classes: {}'.format(cfg.num_classes))
    log_string('** batch_size: {}'.format(cfg.batch_size))
    log_string('** epoch: {}'.format(cfg.epoch))
    log_string('** init_learning_rate: {}'.format(cfg.init_learning_rate))
    log_string('** decay_step: {}'.format(cfg.decay_step))
    log_string('** decay_rate: {}'.format(cfg.decay_rate))
    log_string('** weight_decay: {}'.format(cfg.weight_decay))

    with tf.Graph().as_default():
        anchor, labels = placeholder_inputs(cfg.batch_size, cfg.num_frames)
        input_negative, labels = placeholder_inputs(cfg.batch_size, cfg.num_frames)
        input_positive, labels = placeholder_inputs(cfg.batch_size, cfg.num_frames)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        keep_prob = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, dtype=tf.int64)
        bn_decay = get_bn_decay(global_step)
        tf.summary.scalar('bn_decay', bn_decay)
        bn_decay = True

        # Get model and loss 
        if MODEL_NAME == "simple_ff":
            anchor_embed = build_graph(
                anchor, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay, reuse_layers=False)

            input_positive_embed = build_graph(
                input_positive, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay)

            input_negative_embed = build_graph(
                input_negative, is_training_pl, keep_prob, weight_decay=cfg.weight_decay, bn_decay=bn_decay)
            
            loss, filter_loss = get_loss(anchor_embed, input_positive_embed,
                            input_negative_embed)
        else:
            raise
            
        tf.summary.scalar('total_loss', loss)

        print("\nplaceholders loaded...")
        # raise

        # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
        # correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        # accuracy = correct / float(cfg.batch_size)
        # tf.summary.scalar('accuracy', accuracy)


        # Get training operator
        learning_rate = get_learning_rate(global_step)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        sess = tf.Session(config=config)

        # # restore model #################
        load_model_path = LOGDIR+'/model_epoch_{}'.format(cfg.load_model_epoch)
        try:
          saver = tf.train.Saver()
          saver.restore(sess, load_model_path)
          print("\nPrevious model restored... ", load_model_path)
        except Exception as e:
          print("\nCannot find the requested model... {}".format(e))
          sess.run(tf.global_variables_initializer())
          # %% create a saver object
          saver = tf.train.Saver()
          print("\nCreating new model...", load_model_path)

        if cfg.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        # init = tf.global_variables_initializer()
        # sess.run(init, {is_training_pl: True})
        # saver = tf.train.Saver()
            
        # Plot Variable Histogram
        t_vars = tf.trainable_variables()
        

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOGDIR+'/train')
        train_writer.add_graph(tf.get_default_graph())
        test_writer = tf.summary.FileWriter(LOGDIR+'/test')
        test_writer.add_graph(tf.get_default_graph())
        
        print("\nsession initialized...")

        # Count number of trainable parameters
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        log_string('************ The Number of Trainable Parameters: {} ************'.format(num_params))
        num_g_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()])
        log_string('************ The Number of Global Parameters: {} ************'.format(num_g_params))
        
        ops = {'anchor_pl': anchor,
               'input_neg_pl': input_negative,
               'input_pos_pl': input_positive,
               'is_training_pl': is_training_pl,
               'keep_prob': keep_prob,
               'loss': loss,
               'filter_loss':filter_loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step}
        
        # keep out : [1,3,8,15,18]
        # train with = [0,2,4,5,6,7,9,10,11,12,13,14,16,17,19]
        class_list = [2,4,5,6,7,9,10,11,12,13,14,16,17,19,20]

        training_dataset = np.load(
            'd:/datasets/MSRAction3D/one_shot_train.npy')
        validation_dataset = np.load(
            'd:/datasets/MSRAction3D/one_shot_test_for_known.npy')

        # # load datasets: this is inside the loop to simulate k-fold validation
        # train_dataset, validation_dataset = load_npy_filenames(cfg.dataset_directory+"_"+str(
        #     cfg.num_frames)+"_"+str(cfg.num_points)+"/take-080218/*.npy", validation_split=cfg.validation_split)
        
        train_data_gen = NewTripletGenerator(training_dataset, classes=class_list, batch_size=cfg.batch_size)
        # validation_data_gen = TripletGenerator(validation_dataset, classes=class_list, batch_size=cfg.batch_size)

        for epoch in range(1, cfg.epoch+1):
            log_string('\n******** Training:---Epoch_{}/{} *********'.format(epoch+int(cfg.load_model_epoch), cfg.epoch+int(cfg.load_model_epoch)))
            
            log_string('Training ...')
            train_one_epoch(sess, train_data_gen, ops, train_writer)
            # log_string('Validating ...')
            # val_one_epoch(sess, validation_data_gen, ops, test_writer)

            if epoch % cfg.save_model_freq == 0:
                saver.save(sess, LOGDIR+'/model_epoch_{}'.format(epoch+int(cfg.load_model_epoch)))
                log_string('Model saved at epoch {}'.format(epoch+int(cfg.load_model_epoch)))
                log_string(LOGDIR+'/model_epoch_{}'.format(epoch+int(cfg.load_model_epoch))+'\n')

                
                # log_string('learning rate at epoch {}: {}'.format(epoch, sess.run(model.learning_rate)))

def train_one_epoch(sess, train_data_gen, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    iters_per_epoch = train_data_gen.iters_per_epoch
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    pbar = tqdm(range(iters_per_epoch))

    for iteration in pbar:
        X, current_label = next(train_data_gen.generator)
        # print("\nFeeding...",np.shape(current_data))
        # print("\nFeeding...",iterations)

        # if iteration < 10:
        #     skip = False
        # else:
        X, skip = filter_hard(sess, X, current_label, ops, 
                                    buffer_size=cfg.batch_size)
                                            
        if not skip:

            # for i in range(10): # repeat optimization i times
            feed_dict = {ops['anchor_pl']: X[:,0,:,:],
                            ops['input_neg_pl']: X[:,1,:,:],
                            ops['input_pos_pl']: X[:,2,:,:],
                            ops['is_training_pl']: is_training,
                            ops['keep_prob']: 0.4}

            _, summary, step, loss_val = sess.run([ops['train_op'], ops['merged'], 
                                                            ops['step'], ops['loss']], 
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            
            total_seen += cfg.batch_size
            loss_sum += loss_val
            pbar.set_description("Training Loss %.6f"%(loss_sum/(iteration+1)))
        # else:
        #     print("skipped!!!")
    
    mean_loss = loss_sum / float(iters_per_epoch)
    log_string('mean loss: %f' % (mean_loss))


def filter_hard(sess, X, y, ops, buffer_size=20):
    global BUFFER_SAMPLES
    samples_losses = []
    for sample in X:
      sample = np.expand_dims(sample, axis=0)
    #   print("sample shape: ", np.shape(sample))

      filter_loss = sess.run(ops['filter_loss'], 
            feed_dict={
                    ops['anchor_pl']: sample[:,0,:,:],
                    ops['input_neg_pl']: sample[:,1,:,:],
                    ops['input_pos_pl']: sample[:,2,:,:],
                    ops['is_training_pl']: False,
                    ops['keep_prob']: 1.0
                  })
      
      samples_losses.append(filter_loss)

      # print("shape of sample: ",np.shape(sample))
      # print("filter_loss: ",filter_loss)

    sorted_samples_idx = np.argsort(samples_losses)
    # print("sample losses: ", samples_losses)
    # print("sorted losses indices: ", sorted_samples_idx)

    new_samples_losses = []
    if len(BUFFER_SAMPLES) < buffer_size:
        # print("buffering: ", np.shape(BUFFER_SAMPLES)[0])
        new_samples = BUFFER_SAMPLES
    else:
      BUFFER_SAMPLES = []
      new_samples = []

    for idx in sorted_samples_idx:
      if samples_losses[idx] > 0.0:# and random.random() > 0.5:
        continue
      new_samples_losses.append(samples_losses[idx])
      new_samples.append(X[idx])

    # print("sorted losses: ",np.flip(new_samples_losses,0))
    # print("size of new samples: ", np.shape(new_samples))
    # print("size of buffer samples: ", np.shape(BUFFER_SAMPLES))

    BUFFER_SAMPLES = new_samples
    

    if len(BUFFER_SAMPLES) < buffer_size:
      return None, True
    else:
      
      return np.array(BUFFER_SAMPLES), False


def val_one_epoch(sess, validation_data_gen, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0
    loss_sum = 0
    
    iters_per_epoch = validation_data_gen.iters_per_epoch

    pbar = tqdm(range(iters_per_epoch))

    for iteration in pbar:
        pbar.set_description("Batch {}".format(iteration))

        current_data, current_labels = next(validation_data_gen.generator)

        feed_dict = {ops['anchor_pl']: current_data[:,0,:,:],
                        ops['input_neg_pl']: current_data[:,1,:,:],
                        ops['input_pos_pl']: current_data[:,2,:,:],
                        ops['is_training_pl']: is_training,
                        ops['keep_prob']: 1.0}
        summary, step, loss_val = sess.run([ops['merged'], ops['step'], ops['loss']], 
                                                    feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
 
        total_seen += cfg.batch_size
        loss_sum += loss_val

    total_avg_loss = loss_sum / float(iters_per_epoch)           
    log_string('eval mean loss: %f' % (total_avg_loss))

def save_net():
    model_path = LOGDIR + "/model_epoch_" + cfg.load_model_epoch
    save_name = LOGDIR + "/saved_model_pb"

    #Step 1
    #import the model metagraph
    saver = tf.train.import_meta_graph(model_path+'.meta', clear_devices=True)
    
    # make that as the default graph
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    
    #now restore the variables
    saver.restore(sess, model_path)

    # #Step 2, if output node name is not known
    # # Find the output name
    # graph = tf.get_default_graph()
    # for op in graph.get_operations():
    #   print(op.name)

    #Step 3
    from tensorflow.python.platform import gfile
    from tensorflow.python.framework import graph_util

    output_node_names = "output_node"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        output_node_names.split(","))

    #Step 4
    #output folder
    output_fld = ''
    #output pb file name
    output_model_file = save_name+'.pb'
    from tensorflow.python.framework import graph_io
    #write the graph
    graph_io.write_graph(output_graph_def, output_fld,
                         output_model_file, as_text=False)


def main(_):
    if cfg.is_training:
        print('Start Training ...')
        train()
        print('Finished Training')
        fout.close()
    else:
        save_net()

if __name__ == "__main__":
    tf.app.run()
