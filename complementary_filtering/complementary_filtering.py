import pickle
import argparse
import numpy as np

file1='' # path to TAG proposals after TAR (output of TAR/test_reuslts/post_processing.py)
file2='' # path to SWIN proposals after TAR and PATE (output of PATE/test_results/post_processing.py)

proposal1=pickle.load(open(file1))
proposal2=pickle.load(open(file2))
theta=0.02
alpha=0.2
new_proposal={}
for video in proposal1:
    new_proposal[video]=[]
    count=0
    for k in range(proposal1[video].shape[0]):
        if proposal1[video][k,2]>0.0:
            new_proposal[video].append(proposal1[video][k])
            count+=1
    count=0
    for k in range(proposal2[video].shape[0]):
        if proposal2[video][k,2]>theta:
            prop=proposal2[video][k][0:3]
            prop[2]=prop[2]*(proposal2[video][k,3])*alpha
            new_proposal[video].append(prop)
            count+=1
    new_proposal[video]=np.array(new_proposal[video])
    
    s=new_proposal[video][:,2]
    new_ind=np.argsort(s)[::-1]
    new_proposal[video]=new_proposal[video][new_ind,:]
new_name="./output_pkls/CTAP-results_alpha0.2.pkl"
pickle.dump(new_proposal,open(new_name,"w"))
