""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016
"""

import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, 
                     shape, 
                     initializer, 
                     use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  # with tf.device('/cpu:0'):
  dtype = tf.float16 if use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, 
                                shape, 
                                stddev, 
                                wd, 
                                use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    # tf.add_to_collection('losses', weight_decay)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
  return var

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           use_bias=True,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      if use_bias:
        biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def dilated_conv2d(inputs,
                   output_dims,
                   kernel_size,
                   padding='valid',
                   dilation_rate=[1, 1],
                   activation=tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer,
                   regularizer_scale=0.0,
                   use_bias=True,
                   bn=False,
                   bn_decay=None,
                   name=None,
                   is_training=None,
                   reuse=None):
    kernel_regularizer = kernel_regularizer(regularizer_scale)
    output = tf.layers.conv2d(inputs, 
                              filters=output_dims, 
                              kernel_size=kernel_size,
                              padding=padding, 
                              dilation_rate=dilation_rate,
                              activation=activation, 
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              use_bias=use_bias,
                              name=name,
                              reuse=reuse)
    if bn:
      output = batch_norm_for_conv2d(output, 
                                    is_training=is_training,
                                    bn_decay=bn_decay, 
                                    scope='bn')
    return output

def separable_conv2d(inputs,
               kernel_size,
               channel_multiplier,
               output_dims,
               strides=[1,1,1],
               padding='SAME',
               rate=None,
               stddev=1e-3,
               weight_decay=0.0,
               use_bias=False,
               use_xavier=True,
               bn=False,
               bn_decay=None,
               is_training=None,
               name=None):
    with tf.variable_scope(name) as sc:
        depth_h, depth_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [depth_h, depth_w, num_in_channels, channel_multiplier]
        depth_filter = _variable_with_weight_decay('depth_weights',
                                            shape=kernel_shape,
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
        point_filter_shape = [1, 1, num_in_channels*channel_multiplier, output_dims]
        point_filter = _variable_with_weight_decay('point_weights',
                                            shape=point_filter_shape,
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
        stride_h, stride_w, stride_d = strides
        outputs = tf.nn.separable_conv2d(inputs, depth_filter, point_filter,
                                        [1, stride_h, stride_w, stride_d],
                                        rate=rate, padding=padding)
        if use_bias:
          biases = _variable_on_cpu('biases', [output_dims],
                                  tf.constant_initializer(0.0))
          outputs = tf.nn.bias_add(outputs, biases)

        if bn:
          outputs = batch_norm_for_conv2d(outputs, is_training,
                                          bn_decay=bn_decay, scope='bn')
        return outputs

def depthwise_conv2d(inputs,
                    filter_size,
                    channel_multiplier=1,
                    strides=[1,1,1],
                    padding='SAME',
                    rate=None,
                    stddev=1e-3,
                    weight_decay=0.0,
                    use_bias=False,
                    use_xavier=True,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    name=None):
    with tf.variable_scope(name) as sc:
        depth_h, depth_w = filter_size
        num_in_channels = inputs.get_shape()[-1].value
        depth_filter_shape = [depth_h, depth_w, num_in_channels, channel_multiplier]
        depth_filter = _variable_with_weight_decay('depth_weights',
                                            shape=depth_filter_shape,
                                            use_xavier=use_xavier,
                                            stddev=stddev,
                                            wd=weight_decay)
        stride_h, stride_w, stride_d = strides
        outputs = tf.nn.depthwise_conv2d(inputs, depth_filter,
                                        [1, stride_h, stride_w, stride_d],
                                        padding=padding)
        if use_bias:
          biases = _variable_on_cpu('biases', [num_in_channels * channel_multiplier],
                                  tf.constant_initializer(0.0))
          outputs = tf.nn.bias_add(outputs, biases)

        if bn:
          outputs = batch_norm_for_conv2d(outputs, is_training,
                                          bn_decay=bn_decay, scope='bn')
        return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    use_bias=True,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)

    if use_bias:
      biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def maxout(inputs,
          output_dims,
          scope,
          axis=-1):
  with tf.variable_scope(scope):
    outputs = tf.contrib.layers.maxout(inputs, num_units=output_dims, axis=axis)
  return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 1D convolutional maps.
  
  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)

def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
  """ Batch normalization on 3D convolutional maps.
  
  Args:
      inputs:      Tensor, 5D BDHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2,3], bn_decay)


def batch_norm(inputs,
               is_training):
    return tf.layers.batch_normalization(inputs, training=is_training)

def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
