import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_integer('num_frames', 5, 'number of frames')
flags.DEFINE_integer('num_classes', 20, 'number of classes')

# for training
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('im_dim', 96, 'input_image dimension')
flags.DEFINE_integer('epoch', 100, 'epoch')
flags.DEFINE_float('init_learning_rate', 0.0003, 'initial learning rate')
flags.DEFINE_integer('decay_step', 10*3000, 'decay step') # 20Xdataset_size has worked well
flags.DEFINE_float('decay_rate', 0.9, 'decay rate')
flags.DEFINE_float('weight_decay', 0.001, 'weight_decay')

############################
#   environment setting    #
############################ load_model_epoch
flags.DEFINE_string('model_name', 'simple_ff', 'name of the model') # pointnet or attention
flags.DEFINE_string('logdir', 'logdir_utkinect_2_', 'logs directory')
flags.DEFINE_string('dataset_directory', 'dataset/hand_controls_npy', 'dataset directory')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('load_model_epoch', '0', 'epoch to load model from')
flags.DEFINE_string('test_logdir', 'test_logdir_', 'test logs directory')
flags.DEFINE_integer('save_model_freq', 5, 'the frequency of saving model(epoch)')
flags.DEFINE_boolean('debug', False, 'Debug mode')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
