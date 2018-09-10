import os
import numpy as np

def read_gt(filename,fps_dict):
    op_file = open(filename,"r")
    reading_list={}
    for aline in op_file:
        values = aline.split()
  
        if(reading_list.has_key(values[0])):
            reading_list[values[0]].append((float(values[1])*fps_dict[values[0]],float(values[2])*fps_dict[values[0]]))
        else:
            reading_list[values[0]]= [(float(values[1])*fps_dict[values[0]],float(values[2])*fps_dict[values[0]])]
    op_file.close()
    return reading_list

def read_all_gt(filename_list,fps_dict):
    gt_dict_all={}
    for filename in filename_list:
        gt_dict=read_gt(filename,fps_dict)
        for video in gt_dict:
            if not video in gt_dict_all: gt_dict_all[video]=[]
            gt_dict_all[video].extend(gt_dict[video])
    return gt_dict_all

def calculate_IoU(i0,i1):
    union =(min(i0[0],i1[0]),max(i0[1],i1[1]))
    inter =(max(i0[0],i1[0]),min(i0[1],i1[1]))   
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def generate_swin_clips(swin_list,overlap_list,frame_num):
    all_clip_list=[]
    for k,swin_len in enumerate(swin_list):
        current_pos=1
        while current_pos+swin_len<=frame_num:
            all_clip_list.append((current_pos,current_pos+swin_len))
            current_pos+=int(swin_len*(1-overlap_list[k]))
    return all_clip_list

def read_tag(file_name):
    tag_proposals={}
    with open(file_name) as f:
        for l in f:
            video_name = l.split(" ")[0]
            if not video_name in tag_proposals: tag_proposals[video_name]=[]
            tag_proposals[video_name].append([float(l.split(" ")[1]), float(l.split(" ")[2])])
    return tag_proposals


def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1 + 1.0).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0] + 1) +
                 (target_segments[i, 1] - target_segments[i, 0] + 1) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou


fps_dict={}
length_dict={}
with open("../sample_frames/thumos14_video_length_test.txt") as f:
    for l in f:
        fps_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[2])/float(l.rstrip().split(" ")[1])
        length_dict[l.rstrip().split(" ")[0]]=float(l.rstrip().split(" ")[2])
anno_dir="/home/jiyang/Workspace/Datasets/THUMOS_2014/TH14_Temporal_Annotations_Test/annotation/"
anno_file_list=os.listdir(anno_dir)
for k in range(len(anno_file_list)):
    anno_file_list[k]=anno_dir+anno_file_list[k]

gt_dict=read_all_gt(anno_file_list,fps_dict)
tag_dict=read_tag("/home/jiyang/Workspace/Works/activity_localization/THUMOS_2014/process_tag/tag_proposals_round6.txt")
thresh=0.5
unit_size=6
output_file=open("gt_matched_with_tag_thresh0.5_round6_test.txt","w")

gt_matched_dict={}
for video in gt_dict:
    gt_proposal=np.array(gt_dict[video])
    tag_proposal=np.array(tag_dict[video])
    gt_matched_dict[video]=np.zeros(gt_proposal.shape[0]) 
    iou_mat=segment_tiou(gt_proposal, tag_proposal)
    for i in range(gt_proposal.shape[0]):
        for j in range(tag_proposal.shape[0]):
            if iou_mat[i,j]>thresh:
                gt_matched_dict[video][i]=1 
                break
    print video
    print iou_mat
    print sum(gt_matched_dict[video])
    print gt_matched_dict[video].shape[0]

for video in gt_dict:
    for i in range(len(gt_dict[video])):
        round_start=round(gt_dict[video][i][0]/unit_size)*unit_size+1
        round_end=round(gt_dict[video][i][1]/unit_size)*unit_size+1
        if round_end-round_start>=unit_size and round_end<length_dict[video]:
            output_file.write(video+" "+str(int(round_start))+" "+str(int(round_end))+" "+str(int(gt_matched_dict[video][i]))+"\n")

