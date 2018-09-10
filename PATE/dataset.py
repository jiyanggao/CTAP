
import numpy as np
from math import sqrt
import os
import random
import pickle

def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
A class that handles the training set
'''
class TrainingDataSet(object):
    def __init__(self,flow_feat_dir,appr_feat_dir,clip_gt_path,batch_size,movie_length_info,unit_feature_size,unit_size):
        self.unit_feature_size=unit_feature_size
        self.unit_size=unit_size
        self.batch_size=batch_size
        self.movie_length_info=movie_length_info
        self.visual_feature_dim=self.unit_feature_size
        self.flow_feat_dir=flow_feat_dir
        self.appr_feat_dir=appr_feat_dir
        self.training_samples=[]

        print "Reading training data list from "+clip_gt_path
        with open(clip_gt_path) as f:
            for l in f:
                movie_name=l.rstrip().split(" ")[0]
                clip_start=float(l.rstrip().split(" ")[1])
                clip_end=float(l.rstrip().split(" ")[2])
                label=float(l.rstrip().split(" ")[3])
                self.training_samples.append((movie_name,clip_start,clip_end,label))
        print str(len(self.training_samples))+" training samples are read"
        self.num_samples=len(self.training_samples)
    def calculate_regoffset(self,clip_start,clip_end,round_gt_start,round_gt_end):
        start_offset=(round_gt_start-clip_start)/self.unit_size
        end_offset=(round_gt_end-clip_end)/self.unit_size
        return start_offset, end_offset

    '''
    Get the central features
    '''    
    def get_pooling_feature(self,flow_feat_dir,appr_feat_dir,movie_name,start,end):
        swin_step=self.unit_size
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        current_pos=start
        while current_pos<end:
            swin_start=current_pos
            swin_end=swin_start+swin_step
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat))
            all_feat=np.vstack((all_feat,feat))
            current_pos+=swin_step
        pool_feat=np.max(all_feat,axis=0)
        return pool_feat

    def next_batch(self):

        random_batch_index=random.sample(range(self.num_samples),self.batch_size)
        image_batch=np.zeros([self.batch_size,self.visual_feature_dim])
        label_batch=np.zeros([self.batch_size],dtype=np.int32)
        index=0
        while index < self.batch_size:
                k=random_batch_index[index]
                movie_name=self.training_samples[k][0]
                clip_start=self.training_samples[k][1]
                clip_end=self.training_samples[k][2]
                featmap=self.get_pooling_feature(self.flow_feat_dir, self.appr_feat_dir,movie_name,clip_start,clip_end)
                image_batch[index,:]=featmap
                label_batch[index]=self.training_samples[k][3]
                index+=1
        
        return image_batch, label_batch


'''
A class that handles the test set
'''
class TestingDataSet(object):
    def __init__(self,flow_feat_dir,appr_feat_dir,test_clip_path,batch_size):
        self.batch_size=batch_size
        self.flow_feat_dir=flow_feat_dir
        self.appr_feat_dir=appr_feat_dir
        print "Reading testing data list from "+test_clip_path
        self.test_samples=[]
        with open(test_clip_path) as f:
            for l in f:
                movie_name=l.rstrip().split(" ")[0]
                clip_start=float(l.rstrip().split(" ")[1])
                clip_end=float(l.rstrip().split(" ")[2])
                self.test_samples.append((movie_name,clip_start,clip_end))
        self.num_samples=len(self.test_samples)
        print "test clips number: "+str(len(self.test_samples))
        



