# CTAP
This repository contains tensorflow implementation for *CTAP: Complementary Temporal Action Proposal Generation* in [ECCV 2018](https://arxiv.org/pdf/1807.04821.pdf).

## Setup
The framework of CTAP is shown in the figure below:

<p align="center">
  <img src='img/framework.png' width='900'/>
</p>

This repository mainly consists of two parts: 

- [x] code for Temporal convolutional Adjustment and Ranking (TAR) network which uses temporal conv layers to aggregate the unit-level features 
- [ ] code for Proposal-level Actionness Trustworthiness Estimator (PATE) classifier (`in progress`) 

## Feature downloading

We provide both unit-level features for [TAG](https://github.com/yjxiong/action-detection) scores prediction and sliding windows features for TAR on [THUMOS-14 dataset](http://crcv.ucf.edu/THUMOS14/). 

> Note: validation set is used for training, as the training set for THUMOS-14 does not contain untrimmed videos.

Unit-level features (unit length = 16) downloading links are provided in the table below. 

|            | Appearance | Denseflow |
|:----------:|:----------:|:---------:|
| Validation | [link](https://drive.google.com/file/d/180YUoPvyaF2Z_T9KMKINLdDQCZEg60Jb/view?usp=sharing) | [link](https://drive.google.com/file/d/1-6dmY_Uy-H19HxvfK_wUFQCYHmlPzwFx/view?usp=sharing) |
| Test | [link](https://drive.google.com/file/d/1x9Q78AZiAGqx4XB2zO3SEKp1htsATlnU/view?usp=sharing) | [link](https://drive.google.com/file/d/1Qm9lIJQFm5s6hDSB_2k1tj8q2tnabflJ/view?usp=sharing)|

For Unit-level features (unit length = 6), the appearance and denseflow features for validation and test sets are available in [google drive](http://www.google.com).

## Running TAR
Download the unit level featurs, and edit the feature path in `TAR/main.py`, and then just run  `python main.py`

## Reference

If you find the repository is useful for your research, please consider citing the following work:

```
@inproceedings{gao2018ctap,
  title={CTAP: Complementary Temporal Action Proposal Generation},
  author={Gao*, Jiyang and Chen*, Kan and Nevatia, Ram},
  booktitle={ECCV},
  year={2018}
}
```
