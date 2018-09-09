import tensorflow as tf
import numpy as np
import tar_model
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import vs_multilayer
import operator
import os 

ctx_num=4
central_num=4
unit_size=16.0
unit_feature_size=2048*2
lr=0.005
lambda_reg=2.0
batch_size=128
test_steps=4000



def sample_to_number(all_feats,num):
        sampled_feats=np.zeros([num,all_feats.shape[1]],dtype=np.float32)
        if all_feats.shape[0]==0: return sampled_feats
        if all_feats.shape[0]==num: return all_feats
        else:
            for k in range(num):
                sampled_feats[k]=all_feats[all_feats.shape[0]/num*k,:]
        return sampled_feats


def get_central_feature(flow_feat_dir,appr_feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    current_pos=start
    while current_pos<end:
        swin_start=current_pos
        swin_end=swin_start+swin_step
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy") and os.path.exists(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat))
            all_feat=np.vstack((all_feat,feat))   
        #else: print "not exist: "+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)     
        current_pos+=swin_step
    central_feat=all_feat
    return central_feat


def get_left_context_feature(flow_feat_dir,appr_feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    count=0
    current_pos=start
    context_ext=False
    while  count<ctx_num/2:
        swin_start=current_pos-swin_step
        swin_end=current_pos
        if os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy") and os.path.exists(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat))
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos-=swin_step
        count+=1
    count=0
    current_pos=start
    while  count<ctx_num/2:
        swin_start=current_pos
        swin_end=current_pos+swin_step
        if os.path.exists(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy") and os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat))
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos+=swin_step
        count+=1

    if context_ext:
        pool_feat=all_feat
    else:
        pool_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    return pool_feat


def get_right_context_feature(flow_feat_dir,appr_feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    count=0
    current_pos=end
    context_ext=False
    while  count<ctx_num/2:
        swin_start=current_pos
        swin_end=current_pos+swin_step
        if os.path.exists(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy") and os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat))
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos+=swin_step
        count+=1
    count=0
    current_pos=end
    while  count<ctx_num/2:
        swin_start=current_pos-swin_step
        swin_end=current_pos
        if os.path.exists(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy") and os.path.exists(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
            flow_feat=np.load(flow_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            appr_feat=np.load(appr_feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            flow_feat=flow_feat/np.linalg.norm(flow_feat)
            appr_feat=appr_feat/np.linalg.norm(appr_feat)
            feat=np.hstack((flow_feat,appr_feat)) 
            all_feat=np.vstack((all_feat,feat))
            context_ext=True
        current_pos-=swin_step
        count+=1

    if context_ext:
        pool_feat=all_feat
    else:
        pool_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    return pool_feat


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# test 
def do_eval_slidingclips(sess,vs_eval_op,model,movie_length_info,iter_step):
    results_lst=[]
    results_lst_noreg=[]
    for k,test_sample in enumerate(model.test_set.test_samples):
        if k%1000==0:
            print str(k)+"/"+str(len(model.test_set.test_samples))
        movie_name=test_sample[0]
        movie_length=movie_length_info[movie_name]
        clip_start=test_sample[1]
        clip_end=test_sample[2]
        featmap=get_central_feature(model.test_set.flow_feat_dir,model.test_set.appr_feat_dir,movie_name,clip_start,clip_end)
        left_feat=get_left_context_feature(model.test_set.flow_feat_dir,model.test_set.appr_feat_dir,movie_name,clip_start,clip_end)
        right_feat=get_right_context_feature(model.test_set.flow_feat_dir,model.test_set.appr_feat_dir,movie_name,clip_start,clip_end)
        featmap=sample_to_number(featmap,central_num)
        left_feat=sample_to_number(left_feat,ctx_num)
        right_feat=sample_to_number(right_feat,ctx_num)
        
        feed_dict = {
            model.central_ph_test: np.reshape(featmap, [1, central_num, unit_feature_size]),
            model.left_ph_test: np.reshape(left_feat,[1, ctx_num, unit_feature_size]),
            model.right_ph_test: np.reshape(right_feat,[1, ctx_num, unit_feature_size])
            }
        
        outputs=sess.run(vs_eval_op,feed_dict=feed_dict) 
        reg_end=clip_end+outputs[3]*unit_size
        reg_start=clip_start+outputs[2]*unit_size
        round_reg_end=clip_end+np.round(outputs[3])*unit_size
        round_reg_start=clip_start+np.round(outputs[2])*unit_size
        softmax_score=softmax(outputs[0:2])
        action_score=softmax_score[1] 
        results_lst.append((movie_name,round_reg_start,round_reg_end,reg_start,reg_end,action_score,outputs[0],outputs[1]))
        results_lst_noreg.append((movie_name,round_reg_start,round_reg_end,clip_start,clip_end,action_score,outputs[0],outputs[1]))
    pickle.dump(results_lst,open("./test_results/results_BLA_tconv_twostream_unit6_onSwin5_ctx4_central4_sepnorm_tconv512-512_iter"+str(iter_step)+".pkl","w")) 
    pickle.dump(results_lst_noreg,open("./test_results/results_BLA_tconv_twostream_unit6_onSwin5_ctx4_central4_sepnorm_tconv512-512_noreg_iter"+str(iter_step)+".pkl","w"))

def run_training():
    initial_steps=0
    max_steps=13000
    train_clip_path="val_training_samples.txt"
    background_path="background_samples.txt"
    train_flow_featmap_dir="../../val_fc6_16_overlap0.5_denseflow/"
    train_appr_featmap_dir="../../val_fc6_16_overlap0.5_resnet/"
    test_flow_featmap_dir="../../test_fc6_16_overlap0.5_denseflow/"
    test_appr_featmap_dir="../../test_fc6_16_overlap0.5_resnet/"
    test_clip_path="./test_swin.txt"
    test_video_length_info={}
    with open("./thumos14_video_length_test.txt") as f:
        for l in f:
            test_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])
    train_video_length_info={}
    with open("./thumos14_video_length_val.txt") as f:
        for l in f:
            train_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])

    model=tar_model.TAR_Model(batch_size,train_video_length_info,ctx_num, central_num, unit_feature_size,unit_size,
        lambda_reg,lr,train_clip_path,background_path,test_clip_path,train_flow_featmap_dir,train_appr_featmap_dir,test_flow_featmap_dir,test_appr_featmap_dir)

    with tf.Graph().as_default():
		
        loss_cls_reg,vs_train_op,vs_eval_op, loss_reg=model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)
        sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train_reg()

            _, loss_v, loss_reg_v = sess.run([vs_train_op,loss_cls_reg, loss_reg], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: total loss = %.2f, regression loss = %.2f(%.3f sec)' % (step, loss_v, loss_reg_v, duration)) 

            if (step+1) % test_steps == 0:
                print "Start to test:-----------------\n"
                do_eval_slidingclips(sess,vs_eval_op,model,test_video_length_info,step+1)

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



