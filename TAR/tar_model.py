import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

import vs_multilayer
from util.cnn import fc_layer as fc 
from dataset import TestingDataSet
from dataset import TrainingDataSet


class TAR_Model(object):
    def __init__(self, batch_size,train_video_length_info,ctx_num, central_num, unit_feature_size,unit_size,lambda_reg,lr,train_clip_path,background_path,test_clip_path,train_flow_feature_dir,train_appr_feature_dir,test_flow_feature_dir,test_appr_feature_dir):
        
        self.batch_size = batch_size
        self.test_batch_size=1
        self.lr=lr
        self.lambda_reg=lambda_reg
        self.unit_feature_size=unit_feature_size
        self.visual_feature_dim=unit_feature_size
        self.train_set=TrainingDataSet(train_flow_feature_dir,train_appr_feature_dir,train_clip_path,background_path,batch_size, train_video_length_info,ctx_num, central_num, unit_feature_size,unit_size)
        self.test_set=TestingDataSet(test_flow_feature_dir,test_appr_feature_dir,test_clip_path,self.test_batch_size,ctx_num)
        self.ctx_num= ctx_num
        self.central_num=central_num    	    
    def fill_feed_dict_train_reg(self):
        central_batch, left_batch, right_batch, label_batch,offset_batch=self.train_set.next_batch()
        input_feed = {
                self.central_ph_train: central_batch,
                self.left_ph_train: left_batch,
                self.right_ph_train: right_batch,
                self.label_ph: label_batch,
                self.offset_ph: offset_batch
        }

        return input_feed
            
    # construct the top network and compute loss
    def compute_loss_reg(self,central, start, end ,offsets,labels):
        
        central_cls, start_reg, end_reg=vs_multilayer.vs_multilayer(central, start, end, "BLA")
        offset_pred=tf.concat(1,(start_reg,end_reg))

        #classification loss
        loss_cls_vec=tf.nn.sparse_softmax_cross_entropy_with_logits(central_cls, labels)
        loss_cls=tf.reduce_mean(loss_cls_vec)
        # regression loss
        label_tmp=tf.to_float(tf.reshape(labels,[self.batch_size,1]))
        label_for_reg=tf.concat(1,[label_tmp,label_tmp])
        loss_reg=tf.reduce_mean(tf.mul(tf.abs(tf.sub(offset_pred,offsets)),label_for_reg))

        loss=tf.add(tf.mul(self.lambda_reg,loss_reg),loss_cls)
        return loss,offset_pred,loss_reg


    def init_placeholder(self):
        self.central_ph_train=tf.placeholder(tf.float32, shape=(self.batch_size,self.central_num, self.visual_feature_dim))
        self.left_ph_train=tf.placeholder(tf.float32, shape=(self.batch_size,self.ctx_num, self.visual_feature_dim))
        self.right_ph_train=tf.placeholder(tf.float32, shape=(self.batch_size,self.ctx_num, self.visual_feature_dim))
        self.label_ph=tf.placeholder(tf.int32, shape=(self.batch_size))
        self.offset_ph=tf.placeholder(tf.float32, shape=(self.batch_size,2))
        self.central_ph_test=tf.placeholder(tf.float32, shape=(self.test_batch_size, self.central_num,self.visual_feature_dim))
        self.left_ph_test=tf.placeholder(tf.float32, shape=(self.test_batch_size,self.ctx_num, self.visual_feature_dim))
        self.right_ph_test=tf.placeholder(tf.float32, shape=(self.test_batch_size,self.ctx_num, self.visual_feature_dim))

        return
    

    # set up the eval op
    def eval(self,central, start, end):
        central_cls, start_reg, end_reg=vs_multilayer.vs_multilayer(central, start, end, "BLA", reuse=True)
        outputs=tf.concat(1,(central_cls,start_reg,end_reg))
        outputs=tf.reshape(outputs,[4])
        print "eval output size: " + str(outputs.get_shape().as_list())
        
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
        v_dict=self.get_variables_by_name(["BLA"])
        vs_optimizer=tf.train.AdamOptimizer(self.lr,name='vs_adam')
        vs_train_op=vs_optimizer.minimize(loss,var_list=v_dict["BLA"])
        return vs_train_op

    # construct the network
    def construct_model(self):
        self.init_placeholder()
        self.loss_cls_reg,offset_pred,loss_reg=self.compute_loss_reg(self.central_ph_train,self.left_ph_train, self.right_ph_train,self.offset_ph,self.label_ph)
        self.train_op=self.training(self.loss_cls_reg)
        eval_op=self.eval(self.central_ph_test,self.left_ph_test, self.right_ph_test)
        return self.loss_cls_reg,self.train_op, eval_op,loss_reg


