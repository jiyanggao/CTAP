import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

import vs_multilayer
from util.cnn import fc_layer as fc 
from dataset import TestingDataSet
from dataset import TrainingDataSet


class PATE_Model(object):
    def __init__(self, batch_size, train_video_length_info, unit_feature_size, unit_size, lambda_reg, lr, train_clip_path, test_clip_path, train_flow_feature_dir, train_appr_feature_dir, test_flow_feature_dir, test_appr_feature_dir):
        
        self.batch_size = batch_size
        self.test_batch_size = 1
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.unit_feature_size = unit_feature_size
        self.visual_feature_dim = unit_feature_size
        self.train_set = TrainingDataSet(train_flow_feature_dir,train_appr_feature_dir,train_clip_path, batch_size, train_video_length_info,unit_feature_size,unit_size)
        self.test_set = TestingDataSet(test_flow_feature_dir, test_appr_feature_dir, test_clip_path, self.test_batch_size)
   
    	    
    def fill_feed_dict_train_reg(self):
        image_batch,label_batch=self.train_set.next_batch()
        input_feed = {
                self.visual_featmap_ph_train: image_batch,
                self.label_ph: label_batch
        }

        return input_feed
            
    # construct the top network and compute loss
    def compute_loss(self,visual_feature,labels):

        cls_vec=vs_multilayer.vs_multilayer(visual_feature,"PATE",middle_layer_dim=1000)
        cls_vec=tf.reshape(cls_vec,[self.batch_size,2])

        #classification loss
        loss_cls_vec=tf.nn.sparse_softmax_cross_entropy_with_logits(cls_vec, labels)
        loss_cls=tf.reduce_mean(loss_cls_vec)
        return loss_cls


    def init_placeholder(self):
        visual_featmap_ph_train=tf.placeholder(tf.float32, shape=(self.batch_size,self.visual_feature_dim))
        label_ph=tf.placeholder(tf.int32, shape=(self.batch_size))
        visual_featmap_ph_test=tf.placeholder(tf.float32, shape=(self.test_batch_size,self.visual_feature_dim))

        return visual_featmap_ph_train,visual_featmap_ph_test,label_ph
    

    # set up the eval op
    def eval(self,visual_feature_test):
        outputs=vs_multilayer.vs_multilayer(visual_feature_test,"PATE", middle_layer_dim=1000,reuse=True)
        outputs=tf.reshape(outputs,[2])
        return outputs

    # return all the variables that contains the name in name_list
    def get_variables_by_name(self,name_list):
        v_list=tf.trainable_variables()
        v_dict={}
        for name in name_list:
            v_dict[name]=[]
        for v in v_list:
            for name in name_list:
                if name in v.name: v_dict[name].append(v)

        for name in name_list:
            print "Variables of <"+name+">"
            for v in v_dict[name]:
                print "    "+v.name
        return v_dict

    # set up the optimizer
    def training(self, loss):
        v_dict=self.get_variables_by_name(["PATE"])
        vs_optimizer=tf.train.AdamOptimizer(self.lr,name='vs_adam')
        vs_train_op=vs_optimizer.minimize(loss,var_list=v_dict["PATE"])
        return vs_train_op

    # construct the network
    def construct_model(self):
        self.visual_featmap_ph_train,self.visual_featmap_ph_test,self.label_ph=self.init_placeholder()
        visual_featmap_ph_train_norm=tf.nn.l2_normalize(self.visual_featmap_ph_train,dim=1)
        visual_featmap_ph_test_norm=tf.nn.l2_normalize(self.visual_featmap_ph_test,dim=1)
        self.loss_cls = self.compute_loss(visual_featmap_ph_train_norm,self.label_ph)
        self.train_op=self.training(self.loss_cls)
        eval_op=self.eval(visual_featmap_ph_test_norm)
        return self.loss_cls,self.train_op, eval_op


