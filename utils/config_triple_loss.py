import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_integer('num_points', 1024, 'number of points')
flags.DEFINE_integer('num_frames', 5, 'number of frames')
flags.DEFINE_integer('num_classes', 9, 'number of classes')
# flags.DEFINE_integer('num_segs', 40, 'number of parts for segmentation')
# flags.DEFINE_integer('num_sems', 40, 'number of segments for semantic segmentation')
# for training
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_float('init_learning_rate', 0.0001, 'initial learning rate')
flags.DEFINE_integer('decay_step', 20*50000, 'decay step') # 20Xdataset_size has worked well
flags.DEFINE_float('decay_rate', 0.99, 'decay rate')
flags.DEFINE_float('weight_decay', 0.001, 'weight_decay')
flags.DEFINE_float('validation_split', 0.50, 'validation_split')

############################
#   environment setting    #
############################ load_model_epoch  datasets/new_data_clouds/common_japanese_gestures_1point5_npy_4_2048
flags.DEFINE_string('point_cloud_files_dir', 'hand_controls', 'point cloud files directory')
flags.DEFINE_string('model_name', 'skip_pool', 'name of the model') # pointnet or attention, skip_pool
flags.DEFINE_string('logdir', 'logdir_contrastive_skip_pool_', 'logs directory')
flags.DEFINE_string('dataset_directory',
                    'dataset/hand_controls_npy', 'dataset directory')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_boolean('feat_transform', False, 'use feature transform')
flags.DEFINE_string('load_model_epoch', '0', 'epoch to load model from')
flags.DEFINE_string('test_logdir', 'test_logdir_', 'test logs directory')
flags.DEFINE_integer('save_model_freq', 5, 'the frequency of saving model(epoch)')
flags.DEFINE_integer('aug_batches', 2, 'the number of the augmented batches')
flags.DEFINE_boolean('debug', False, 'Debug mode')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
