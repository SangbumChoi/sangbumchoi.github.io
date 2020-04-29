---
title: 'OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields'
date: 2020-04-28
permalink: /posts/2020/04/OpenPose/
tags:
  - Paper
  - Computer Vision
  - Pose Estimation
---

**Introduction**

Normally, most papers were focused on detect or find individuals part in multi human pose. However, this paper shows an effecient method for multi-person pose estimation by using part affinity fields (PAFs)

![Overall pipline](/images/Openpose/Overall_Figure.PNG)

**Related Work**

1. single person pose estimation

CNN is widely used

2. multi person pose estimation

Top-down method, which is detecting single person from multi crowded situation. However, it is inevitable to avoid global inference.   

**Method**

3.1 network architecture

* Using initial 10 layers of VGG-19 and fine-tuned
* replacing 7/7 kernel into 3/3 3 kernels

3.2 simultaneous detection and association

![Overall pipline](/images/Openpose/Overall_Network.PNG)

$$ L_t = phi^t(F,L^(t-1)), 2 <= t <=T_P $$ this refers to front stage of building affinity field

loss is calculated by using L2 distance between ground truth

3.3 confidence maps for part detection

3.4 part affinity fields for part association

3.5 multi-person parsing using pafs

![Overall pipline](/images/Openpose/Ground_Truth_Affinity_Field.PNG)

**Openpose**

body, foot, hand, and facial keypoints on single images

22FPS in a machine with a Nvidia GTX 1080 Ti

**Datasets and Evaluations**

1. MP2 dataset

2. COCO keypoints challenge

3. inference runtime analysis

CNN processing time complexity is O(1), varying with number of people. O(n^2) time complexity due to number of people n.

![Overall pipline](/images/Openpose/Inference_Time.PNG)

GTX-1080 Ti, CPU with i7-6850K. Interesting point is body+foot model has X2 faster than original result. However CPU takes X5 times slower.

**Conclusion**

Reference

[1] https://arxiv.org/pdf/1802.00977.pdf 

[2] https://arxiv.org/pdf/1712.09184.pdf 

[3] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

[4] https://motchallenge.net/results/3D_MOT_2015/?chl=3&orderBy=MOTA&orderStyle=DESC&det=Public

[5] https://arxiv.org/pdf/1612.00137.pdf

GitHub

[1] https://github.com/YuliangXiu/PoseFlow
