import os
import sys
import numpy as np
import math
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils.ops import *
from utils.config import cfg
from utils.data_generator_utils import DataGenerator, load_npy_filenames, jitter_point_cloud

from tqdm import tqdm
from sklearn.utils import shuffle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

MODEL_NAME = cfg.model_name

from motion_model import build_graph, get_loss

LOGDIR = cfg.logdir+MODEL_NAME+"_"+str(cfg.num_frames)+"_"+str(cfg.im_dim)
if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)

log_screen_file = LOGDIR + '/log_screen.txt'
if os.path.exists(log_screen_file):
    os.remove(log_screen_file)

fout = open(log_screen_file,'w')

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
        cfg.batch_size, 240, 320, num_frames))
    labels_pl = tf.placeholder(tf.int32, shape=(None))
    return inputs_pl, labels_pl

def train():
    log_string('***** Config *****')
    log_string('***** Building Point {}...'.format(MODEL_NAME))
    log_string('** num_frames: {}'.format(cfg.num_frames))
    log_string('** num_classes: {}'.format(cfg.num_classes))
    log_string('** batch_size: {}'.format(cfg.batch_size))
    log_string('** epoch: {}'.format(cfg.epoch))
    log_string('** init_learning_rate: {}'.format(cfg.init_learning_rate))
    log_string('** decay_step: {}'.format(cfg.decay_step))
    log_string('** decay_rate: {}'.format(cfg.decay_rate))
    log_string('** weight_decay: {}'.format(cfg.weight_decay))

    with tf.Graph().as_default():
        inputs, labels = placeholder_inputs(cfg.batch_size, cfg.num_frames)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        keep_prob_pl = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, dtype=tf.int64)
        bn_decay = get_bn_decay(global_step)
        tf.summary.scalar('bn_decay', bn_decay)

        pred = build_graph(inputs, is_training_pl, weight_decay=cfg.weight_decay, 
                keep_prob=keep_prob_pl, bn_decay=bn_decay)
        loss = get_loss(pred, labels)

        # raise
        
            
        tf.summary.scalar('total_loss', loss)

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
        correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        accuracy = correct / float(cfg.batch_size)
        tf.summary.scalar('accuracy', accuracy)


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
        # config.log_device_placement = True
        # config.gpu_options.allocator_type = "BFC"
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
        
        # running_vars = tf.get_collection('metric_vars')
        # running_vars = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
        # running_vars = [ var for var in running_vars if isinstance(var, tf.Variable)]
        # print(running_vars)
        # running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        

        # Count number of trainable parameters
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        log_string('************ The Number of Trainable Parameters: {} ************'.format(num_params))
        num_g_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()])
        log_string('************ The Number of Global Parameters: {} ************'.format(num_g_params))
        
        ops = {'inputs_pl': inputs,
               'labels_pl': labels,
               'is_training_pl': is_training_pl,
               'keep_prob_pl':keep_prob_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step}
        

        training_dataset = np.load(
            '/media/tjosh/vault/MSRAction3D/pc_npy_5_training.npy')
        validation_dataset = np.load(
            '/media/tjosh/vault/MSRAction3D/pc_npy_5_test.npy')
        # set_size = len(dataset)
        # dataset = shuffle(dataset)
        # training_dataset = dataset[:int(set_size*0.67)]
        # validation_dataset = dataset[int(set_size*0.67):]
        validation_dataset = validation_dataset[:10000]
        
        train_data_gen = DataGenerator(training_dataset, batch_size=cfg.batch_size)
        validation_data_gen = DataGenerator(validation_dataset, batch_size=cfg.batch_size, augment=False)

        for epoch in range(1, cfg.epoch+1):
            log_string('\n******** Training:---Epoch_{}/{} *********'.format(epoch, cfg.epoch))
            
            log_string('Training ...')
            train_one_epoch(sess, train_data_gen, ops, train_writer)
            log_string('Validating ...')
            val_one_epoch(sess, validation_data_gen, ops, test_writer)

            if epoch % cfg.save_model_freq == 0:
                saver.save(sess, LOGDIR+'/model_epoch_{}'.format(epoch))
                log_string('Model saved at epoch {}'.format(epoch))

                
def test_model():

    with tf.Graph().as_default():

        validation_dataset = np.load('path/to/dataset')
        validation_data_gen = DataGenerator(
            validation_dataset, batch_size=cfg.batch_size)

        inputs, labels = placeholder_inputs(
            cfg.batch_size, cfg.num_frames)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        keep_prob_pl = tf.placeholder(tf.float32)

        global_step = tf.Variable(0, dtype=tf.int64)
        bn_decay = get_bn_decay(global_step)
        tf.summary.scalar('bn_decay', bn_decay)

        # Get model and loss
        pred = build_graph(inputs, is_training_pl, weight_decay=cfg.weight_decay,
                           keep_prob=keep_prob_pl, bn_decay=bn_decay)
        loss = get_loss(pred, labels)

        tf.summary.scalar('total_loss', loss)
        # raise

        correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
        correct = tf.reduce_sum(tf.cast(correct, tf.float32))
        accuracy = correct / float(cfg.batch_size)
        tf.summary.scalar('accuracy', accuracy)

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
        # config.log_device_placement = True
        # config.gpu_options.allocator_type = "BFC"
        sess = tf.Session(config=config)

        # # restore model #################
        # load_model_path = LOGDIR+'/model_epoch_{}'.format(cfg.load_model_epoch)
        load_model_path = LOGDIR+'/model_epoch_{}'.format(cfg.load_model_epoch)
        try:
          saver = tf.train.Saver()
          saver.restore(sess, load_model_path)
          print("\nLoaded previous model... ", load_model_path)
        except Exception as e:
          raise

        if cfg.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Plot Variable Histogram
        t_vars = tf.trainable_variables()
        # for var in t_vars:
        #     tf.summary.histogram(var.op.name, var)

        # saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOGDIR+'/train')
        train_writer.add_graph(tf.get_default_graph())
        test_writer = tf.summary.FileWriter(LOGDIR+'/test')
        test_writer.add_graph(tf.get_default_graph())

        # Count number of trainable parameters
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print('************ The Number of Trainable Parameters: {} ************'.format(num_params))
        num_g_params = np.sum([np.prod(v.get_shape().as_list())
                               for v in tf.global_variables()])
        print('************ The Number of Global Parameters: {} ************'.format(num_g_params))

        ops = {'inputs_pl': inputs,
               'labels_pl': labels,
               'keep_prob_pl': keep_prob_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step}

        # validation_data_gen = DataGenerator(validation_dataset, batch_size=cfg.batch_size)

        print('Validating ...')
        val_one_epoch(sess, validation_data_gen, ops,
                      test_writer, logging=False)

        # log_string('learning rate at epoch {}: {}'.format(epoch, sess.run(model.learning_rate)))


def train_one_epoch(sess, train_data_gen, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    iters_per_epoch = train_data_gen.iters_per_epoch

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    pbar = tqdm(range(iters_per_epoch))

    # for batch_idx in range(num_batches):
    for iteration in pbar:
        current_data, current_label = next(train_data_gen.generator)

        # # Augment batched point clouds by jittering
        # if np.random.rand()>0.3:
        #     current_data = jitter_point_cloud(current_data, sigma=200)

        feed_dict = {ops['inputs_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['keep_prob_pl']: 0.4,
                     ops['is_training_pl']: is_training}
        _, summary, step, loss_val, pred_val = sess.run([ops['train_op'], ops['merged'],
                                                         ops['step'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # print("predicted: ", pred_val)
        pred_val = np.argmax(pred_val, 1)
        # print("predicted: ", pred_val)
        # print("correct: ", current_label)
        correct = np.sum(pred_val == current_label)
        total_correct += correct
        total_seen += cfg.batch_size
        loss_sum += loss_val
        pbar.set_description("Training Accuracry: %.6f,  Training Loss: %.6f" % (
            (total_correct/(total_seen)), loss_sum/(iteration+1)))
        # print("correct: ", correct)

    mean_loss = loss_sum / float(iters_per_epoch)
    mean_acc = total_correct / float(total_seen)
    # precision, recall, TP, TN, FP, FN = sess.run([ops['precision'], ops['recall'], ops['TP'],
    #                                              ops['TN'], ops['FP'], ops['FN']])
    # f1_score = F1_score(precision, recall)
    # mcc = MCC(float(TP),float(TN),float(FP),float(FN))
    # precision, recall, f1_score = sess.run([ops['precision'], ops['recall'], ops['f1_score']])
    log_string('mean loss: %f' % (mean_loss))
    log_string('accuracy: %f' % (mean_acc))

    # tf.saved_model.simple_save(sess, LOGDIR+'/saved_model_simple', inputs={"input_node": ops['inputs_pl']}, outputs={"output":ops['pred']})
    # log_string('precision: %f' % (precision))
    # log_string('recall: %f' % (recall))
    # log_string('f1_score: %f' % (f1_score))


def val_one_epoch(sess, validation_data_gen, ops, test_writer, logging=True):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(cfg.num_classes)]
    total_correct_class = [0 for _ in range(cfg.num_classes)]

    iters_per_epoch = validation_data_gen.iters_per_epoch

    # for batch_idx in range(num_batches):
    for iterations in range(iters_per_epoch):
        current_data, current_labels = next(validation_data_gen.generator)

        feed_dict = {ops['inputs_pl']: current_data,
                     ops['labels_pl']: current_labels,
                     ops['keep_prob_pl']: 1.0,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                                     feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_labels)
        total_correct += correct
        total_seen += cfg.batch_size
        loss_sum += loss_val
        # print("correct: ", correct)
        # for i in range(cfg.batch_size):
        #     l = current_labels[i]
        #     total_seen_class[l] += 1
        #     total_correct_class[l] += (pred_val[i] == l)

    total_avg_loss = loss_sum / float(iters_per_epoch)
    total_avg_acc = total_correct / float(total_seen)
    if logging:
        log_string('eval mean loss: %f' % (total_avg_loss))
        log_string('eval accuracy: %f' % (total_avg_acc))
    else:
        print('eval mean loss: %f' % (total_avg_loss))
        print('eval accuracy: %f' % (total_avg_acc))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))


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
        # test_model()
    else:
        save_net()

if __name__ == "__main__":
    tf.app.run()
