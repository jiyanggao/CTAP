from __future__ import division

import numpy as np
import tensorflow as tf

# components

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_1d_layer as conv_1d
from util.cnn import conv_1d_relu_layer as conv_relu_1d
from util.cnn import conv_layer as conv_2d
from util.cnn import conv_relu_layer as conv_relu_2d
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util.cnn import pooling_layer_1d as max_pool_1d

def vs_multilayer(central_batch, start_batch, end_batch, name,middle_layer_dim=1000, output_size=2, reuse=False):
    conv_kernel=3
    max_pool_kernel=2
    max_pool_stride=2

    with tf.variable_scope(name+"-central"):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"
        
        central_tconv1=conv_relu_1d("temoral_conv1",central_batch,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv1 "+str(central_tconv1.get_shape().as_list())
        central_tconv1=max_pool_1d("max_pool1",central_tconv1,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool1 "+str(central_tconv1.get_shape().as_list())
        central_tconv2=conv_relu_1d("temoral_conv2",central_tconv1,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv2 "+str(central_tconv2.get_shape().as_list())
        central_tconv2=max_pool_1d("max_pool2",central_tconv2,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool2 "+str(central_tconv2.get_shape().as_list())
        central_outputs = fc('layer2', central_tconv2, output_dim=2)

    with tf.variable_scope(name+"-start"):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"
        
        start_tconv1=conv_relu_1d("temoral_conv1",start_batch,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv1 "+str(start_tconv1.get_shape().as_list())
        start_tconv1=max_pool_1d("max_pool1",start_tconv1,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool1 "+str(start_tconv1.get_shape().as_list())
        start_tconv2=conv_relu_1d("temoral_conv2",start_tconv1,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv2 "+str(start_tconv2.get_shape().as_list())
        start_tconv2=max_pool_1d("max_pool2",start_tconv2,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool2 "+str(start_tconv2.get_shape().as_list())
        start_outputs = fc('layer2', start_tconv2, output_dim=1)

    with tf.variable_scope(name+"-end"):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"

        end_tconv1=conv_relu_1d("temoral_conv1",end_batch,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv1 "+str(end_tconv1.get_shape().as_list())
        end_tconv1=max_pool_1d("max_pool1",end_tconv1,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool1 "+str(end_tconv1.get_shape().as_list())
        end_tconv2=conv_relu_1d("temoral_conv2",end_tconv1,kernel_size=conv_kernel,stride=1,output_dim=512)
        print "temporal conv2 "+str(end_tconv2.get_shape().as_list())
        end_tconv2=max_pool_1d("max_pool2",end_tconv2,kernel_size=max_pool_kernel,stride=max_pool_stride)
        print "temporal maxpool2 "+str(end_tconv2.get_shape().as_list())
        end_outputs = fc('layer2', end_tconv2, output_dim=1)
    return central_outputs, start_outputs, end_outputs



