---
title: 'OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields'
title_ko: 'OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields 리뷰'
date: 2020-04-28
permalink: /posts/2020/04/OpenPose/
tags:
  - Paper
  - Computer Vision
  - Pose Estimation
---

<div class="lang-block" lang="en" markdown="1">

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

</div>

<div class="lang-block" lang="ko" markdown="1">

**서론**

일반적으로 대부분의 논문은 multi human pose에서 개별 part를 detect하거나 찾는 데 집중했습니다. 그러나 이 논문은 part affinity fields (PAFs)를 사용한 효율적인 multi-person pose estimation 방법을 제시합니다.

![전체 pipeline](/images/Openpose/Overall_Figure.PNG)

**관련 연구**

1. single person pose estimation

CNN이 널리 사용됩니다.

2. multi person pose estimation

Top-down 방식은 여러 사람이 밀집한 상황에서 각 사람을 detect하는 방식입니다. 그러나 global inference를 피할 수 없습니다.   

**방법**

3.1 네트워크 구조

* VGG-19의 초기 10개 layer를 사용하고 fine-tuning합니다.
* 7/7 kernel을 3개의 3/3 kernel로 대체합니다.

3.2 동시 detection 및 association

![전체 pipeline](/images/Openpose/Overall_Network.PNG)

$$ L_t = phi^t(F,L^(t-1)), 2 <= t <=T_P $$ 이는 affinity field를 구축하는 앞단 stage를 나타냅니다.

Loss는 ground truth와의 L2 distance를 사용하여 계산합니다.

3.3 part detection을 위한 confidence map

3.4 part association을 위한 part affinity field

3.5 PAF를 이용한 multi-person parsing

![전체 pipeline](/images/Openpose/Ground_Truth_Affinity_Field.PNG)

**Openpose**

단일 이미지에서 body, foot, hand, facial keypoint를 추정합니다.

Nvidia GTX 1080 Ti 머신에서 22FPS

**데이터셋 및 평가**

1. MP2 dataset

2. COCO keypoints challenge

3. inference runtime 분석

CNN 처리 시간 복잡도는 사람 수와 관계없이 O(1)입니다. 사람 수 n에 따라서는 O(n^2)의 시간 복잡도를 가집니다.

![전체 pipeline](/images/Openpose/Inference_Time.PNG)

GTX-1080 Ti, i7-6850K CPU 환경입니다. 흥미로운 점은 body+foot 모델이 원래 결과보다 X2 빠르다는 것입니다. 그러나 CPU에서는 X5배 느립니다.

**결론**

참고 문헌

[1] https://arxiv.org/pdf/1802.00977.pdf 

[2] https://arxiv.org/pdf/1712.09184.pdf 

[3] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

[4] https://motchallenge.net/results/3D_MOT_2015/?chl=3&orderBy=MOTA&orderStyle=DESC&det=Public

[5] https://arxiv.org/pdf/1612.00137.pdf

GitHub

[1] https://github.com/YuliangXiu/PoseFlow

</div>
