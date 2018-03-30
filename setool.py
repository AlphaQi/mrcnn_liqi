# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


def batch_norm(x, conv=True, phase_train=False, name=None, data_dict=None):
    epsilon = 2e-5
    # 0, 1, 2 indicates conv
    # tf.nn.moments(x, [0]) non-conv
    if conv:
      list_m = [0, 1, 2]
    else:
      list_m = [0]
    batch_mean_im, batch_var_im = tf.nn.moments(x, list_m)
    with tf.name_scope(name) as scope:
        beta = tf.Variable(tf.constant(0.0, shape=[x.get_shape().as_list()[-1]]),
                             name=scope+"beta", trainable=phase_train)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.get_shape().as_list()[-1]]),
                              name=scope+'gamma', trainable=phase_train)
        batch_mean = tf.Variable(tf.constant(0.0, shape=[x.get_shape().as_list()[-1]]),
                             name=scope+"moving_mean", trainable=phase_train)
        batch_var = tf.Variable(tf.constant(1.0, shape=[x.get_shape().as_list()[-1]]),
                              name=scope+'moving_variance', trainable=phase_train)
        #training 
        # batch_mean = batch_mean_im
        # batch_var = batch_var_im
        if data_dict is not None:
            if scope+"beta:0" in data_dict.keys():
                beta = tf.assign(beta,data_dict[scope+"beta:0"])
                gamma = tf.assign(gamma,data_dict[scope+"gamma:0"])
                batch_mean = tf.assign(batch_mean,data_dict[scope+"moving_mean:0"])
                batch_var = tf.assign(batch_var,data_dict[scope+"moving_variance:0"])
        # inference
        # if data_dict is not None:
        #     if beta.name in data_dict.keys():
        #         beta = tf.assign(beta,data_dict[beta.name])
        #         gamma = tf.assign(gamma,data_dict[gamma.name])
        #         batch_mean = tf.assign(batch_mean,data_dict[scope+"moving_mean:0"])
        #         batch_var = tf.assign(batch_var,data_dict[scope+"moving_variance:0"])
        if "Assign" in beta.name:
            pass
        else:
            print(beta.name)
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=beta,
                                  scale=gamma,
                                  name=name,
                                  variance_epsilon=epsilon)
    return x

def conv_op(input_op, name, padding='SAME', kh=3, kw=3, n_out=1, dh=1, dw=1, data_dict=None, bnflag=False, is_training=False):
    '''
    Args:
    input_op：输入的tensor
    name：这一层的名称
    kh：kernel height即卷积核的高
    kw：kernel weight即卷积核的宽
    n_out：卷积核数量即输出通道数
    dh：步长的高
    dw：步长的宽
    p：参数列表
    '''
    n_in = input_op.get_shape()[-1].value #  get the channel
    with tf.name_scope(name) as scope: 
        kernel = tf.get_variable(scope+"w",  #  use tf.get_variable create kernel 
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d()) # initial
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32) # biases
        biases = tf.Variable(bias_init_val, trainable=True, name='b')  
        if data_dict is not None:
            if kernel.name in data_dict.keys():
                kernel = tf.assign(kernel,data_dict[kernel.name])
            if biases.name in data_dict.keys():
                biases = tf.assign(biases,data_dict[biases.name])

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding=padding)
        # after call the assign , the name has changed
        if "Assign" in kernel.name:
            pass
        else:
            print(kernel.name, kernel.shape)
            print(biases.name, biases.shape)
        z = tf.nn.bias_add(conv, biases)
        return z

def conv_rpn(inputs_list, name, padding='SAME', kernel_size=3, output_num=1, stride=1, data_dict=None):
    '''
    Args:
    input_op：输入的tensor list
    name：这一层的名称
    '''
    output_list = []
    n_in = inputs_list[0].get_shape()[-1].value 

    with tf.name_scope(name) as scope: 
        kernel = tf.get_variable(scope+"w",
                                 shape=[kernel_size, kernel_size, n_in, output_num], 
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias_init_val = tf.constant(0.0, shape=[output_num], dtype=tf.float32) 
        biases = tf.Variable(bias_init_val, trainable=True, name='b') 
        if data_dict is not None:
            if kernel.name in data_dict.keys() and kernel.shape == data_dict[kernel.name].shape:
                kernel = tf.assign(kernel,data_dict[kernel.name])
            if biases.name in data_dict.keys() and biases.shape == data_dict[biases.name].shape:
                biases = tf.assign(biases,data_dict[biases.name])

        for maps in range(len(inputs_list)):
            conv = tf.nn.conv2d(inputs_list[maps], kernel, (1, stride, stride, 1), padding=padding)
            conv_out = tf.nn.bias_add(conv, biases) 
            # print("conv iyt  ", inputs_list[0].shape)
            output_list.append(conv_out)
        if "Assign" in kernel.name:
            pass
        else:
            print(kernel.name, kernel.shape)
            print(biases.name, biases.shape)
        return output_list 

def de_conv_op(input_op, name, output_shape, padding='SAME', kernel_size=2, stride=2, output_num=1, data_dict=None):
    '''
    Args:
    input_op：输入的tensor
    name：这一层的名称
    kernel_size 即卷积核的高
    output_num
    dh：步长的高
    dw：步长的宽
    p：参数列表
    '''
    in_number = input_op.get_shape()[-1].value # 获取input_op的通道数

    with tf.name_scope(name) as scope: # 设置scope，生成的Variable使用默认的命名
        kernel = tf.get_variable(scope+"w",  # kernel（即卷积核参数）使用tf.get_variable创建
                                 shape=[kernel_size, kernel_size, in_number, output_num], # 【卷积核的高，卷积核的宽、输入通道数，输出通道数】
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d()) # 参数初始化
        bias_init_val = tf.constant(0.0, shape=[output_num], dtype=tf.float32) # biases使用tf.constant赋值为0
        biases = tf.Variable(bias_init_val, trainable=True, name='b') # 将bias_init_val转成可训练的参数
        if data_dict is not None:
            if kernel.name in data_dict.keys():
                kernel = tf.assign(kernel,data_dict[kernel.name])
                pass
            if biases.name in data_dict.keys():
                biases = tf.assign(biases,data_dict[biases.name])
                pass
        # 使用tf.nn.conv2d对input_op进行卷积处理，卷积核kernel，步长dh*dw，padding模式为SAME

        conv = tf.nn.conv2d_transpose(input_op, kernel, output_shape,\
                               strides=[1,stride,stride,1], padding='SAME',name=name)
        z = tf.nn.bias_add(conv, biases) # 将卷积结果conv和bias相加
        if "Assign" in kernel.name:
            pass
        else:
            print(kernel.name, kernel.shape)
            print(biases.name, biases.shape)
        return z # 将卷积层的输出activation作为函数结果返回

def fc_op(input_op, name, n_out, data_dict=None, is_training=False):  

    n_in = input_op.get_shape()[-1].value 
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", 
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        # biases赋值0.1以避免dead neuron
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b') 

        if data_dict is not None:
            if kernel.name in data_dict.keys():
                kernel = tf.assign(kernel, data_dict[kernel.name])
            if biases.name in data_dict.keys():
                biases = tf.assign(biases, data_dict[biases.name]) 
        activation = tf.nn.relu(tf.matmul(input_op, kernel) + biases, name=scope) 
        if "Assign" in kernel.name:
            pass
        else:
            print(kernel.name, kernel.shape)
            print(biases.name, biases.shape)
        return activation

def mpool_op(input_tensor=None, k=2, s=2, padding='SAME',name=None): 
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, k, k, 1],
                          strides=[1, s, s, 1], 
                          padding=padding,
                          name=name
                          )