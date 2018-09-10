import pickle
import numpy as  np
import random
from operator import itemgetter
import sys




def compute_prob_dist(clip_length_file):

    length_dist = {}
    for _key in [16,32,64,128,256,512]:
        length_dist[_key] = 0
    with open(clip_length_file) as f:
        for line in f:
            clip_length = int(line.split(" ")[2])-int(line.split(" ")[1])
            length_dist[clip_length] += 1
    sample_sum = sum([length_dist[_key] for _key in length_dist])
    prob = [float(length_dist[_key])/sample_sum for _key in [16,32,64,128,256,512]]
    return prob


pkl_file = "../../TAR/test_results/"+sys.argv[1]
notTAG_file=sys.argv[2]
result_dict = {}
result_dict_sliding = {}
results = pickle.load(open(pkl_file))
notTAG_scores=pickle.load(open(notTAG_file))

clip_length_file = "../../TAR/val_training_samples.txt"
clip_prob=compute_prob_dist(clip_length_file)

for k in range(len(results)):
    result=results[k]
    not_score=notTAG_scores[k][3]
    movie_name = result[0]
    sliding_start = result[1]/30.0
    sliding_end = result[2]/30.0
    reg_start = result[3]/30.0
    reg_end = result[4]/30.0
    conf = result[5]

    if movie_name not in result_dict:
        result_dict[movie_name] = [[reg_start,reg_end,conf, not_score]]
        result_dict_sliding[movie_name] = [[sliding_start,sliding_end,conf, not_score]]
    else:
        result_dict[movie_name].append([reg_start,reg_end,conf,not_score])
        result_dict_sliding[movie_name].append([sliding_start,sliding_end,conf, not_score])

for _key in result_dict:
    result_dict[_key] = sorted(result_dict[_key],key=itemgetter(2))[::-1]
    result_dict[_key] = np.array(result_dict[_key])
    
    x1 = result_dict[_key][:,0]
    x2 = result_dict[_key][:,1]
    s = result_dict[_key][:,2]
    for k in range(x1.shape[0]):
        clip_length_index=[16,32,64,128,256,512].index(min([16,32,64,128,256,512],key=lambda x:abs(x-int(x2[k]*30-x1[k]*30))))
        s[k] = s[k]*clip_prob[clip_length_index]
    new_ind=np.argsort(s)[::-1]
    result_dict[_key] = result_dict[_key][new_ind,:]
    
pickle.dump(result_dict,open("./after_postprocessing/"+sys.argv[1].split(".pkl")[0]+notTAG_file,"wb"))
