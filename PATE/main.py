import tensorflow as tf
import numpy as np
import pate_model
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import vs_multilayer
import operator
import os 

ctx_num=4
unit_size=6.0
unit_feature_size=2048*2
lr=0.005
lambda_reg=2.0
batch_size=128
test_steps=200

def get_pooling_feature(flow_feat_dir, appr_feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
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
    pool_feat=np.mean(all_feat,axis=0)
    return pool_feat


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# test 
def do_eval_slidingclips(sess,vs_eval_op,model,movie_length_info,iter_step):
    results_lst=[]
    for k,test_sample in enumerate(model.test_set.test_samples):
        if k%1000==0:
            print str(k)+"/"+str(len(model.test_set.test_samples))
        movie_name=test_sample[0]
        movie_length=movie_length_info[movie_name]
        clip_start=test_sample[1]
        clip_end=test_sample[2]
        featmap=get_pooling_feature(model.test_set.flow_feat_dir,model.test_set.appr_feat_dir,movie_name,clip_start,clip_end)
        feat=featmap
        feat=np.reshape(feat,[1,unit_feature_size])
        
        feed_dict = {
            model.visual_featmap_ph_test: feat
            }
        
        outputs=sess.run(vs_eval_op,feed_dict=feed_dict)
        softmax_score=softmax(outputs) 
        notTAG_score=softmax_score[0] 
        results_lst.append((movie_name,clip_start,clip_end,notTAG_score))
    pickle.dump(results_lst,open("./test_results/results_twostream_thresh0.5_unit6_onSWIN_iter"+str(iter_step)+".pkl","w")) 

def run_training():
    initial_steps=0
    max_steps=1001
    train_clip_path="gt_matched_with_tag_thresh0.5_round6.txt"
    train_flow_featmap_dir="../../val_fc6_6_overlap0_denseflow/"
    train_appr_featmap_dir="../../val_fc6_6_overlap0_resnet/"
    test_flow_featmap_dir="../../test_fc6_6_overlap0_denseflow/"
    test_appr_featmap_dir="../../test_fc6_6_overlap0_resnet/"
    test_clip_path="./test_swin_unit6_sample4.txt"
    test_video_length_info={}
    with open("./thumos14_video_length_test.txt") as f:
        for l in f:
            test_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])
    train_video_length_info={}
    with open("./thumos14_video_length_val.txt") as f:
        for l in f:
            train_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])

    model=pate_model.PATE_Model(batch_size,train_video_length_info,unit_feature_size,unit_size,
        lambda_reg,lr,train_clip_path,test_clip_path,train_flow_featmap_dir,train_appr_featmap_dir,test_flow_featmap_dir,test_appr_featmap_dir)

    with tf.Graph().as_default():
		
        loss,vs_train_op,vs_eval_op=model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)
        sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train_reg()

            _, loss_v= sess.run([vs_train_op,loss,], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: total loss = %.2f(%.3f sec)' % (step, loss_v, duration)) 

            if (step+1) % test_steps == 0:
                print "Start to test:-----------------\n"
                do_eval_slidingclips(sess,vs_eval_op,model,test_video_length_info,step+1)

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



