---
title: 'PoseFlow: Efficient Online Pose Tracking'
title_ko: 'PoseFlow: Efficient Online Pose Tracking 리뷰'
date: 2020-01-09
permalink: /posts/2020/01/PoseFlow Efficient Online Pose Tracking/
tags:
  - Paper
  - Computer Vision
  - Object Tracking
---

<div class="lang-block" lang="en" markdown="1">

Basically estimation and tracking is easy terminology by understanding definition of respective keywords. Additionally, in a viewpoint of each results, both may same exactly although have different meaning. There are two methods of tracking human pose, one is top-down, the other is bottom-up. In my comprehension estimation is usually used in an image and tracking is used in a video.

Top-down method is basically detecting boxes first and annotate the keypoints with connecting line to draw like pose in the boxes. Bottom-up method is detecting all the keypoints first and showing the result.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*DMdb6SwPEeQBvqbFF6bXNg.jpeg" width="40%">.
</p>

According to [Pose Flow: Efficient Online Pose Tracking from Shanghai Jiao Tong University], top-down method is much more effective in both accuracy and tracking speed. In a definition of accuracy, there are two checking systems, which are mAP(mean average precision) and MOTA(multiple object tracking accuracy, not precision) respectively. 

New Terminology
1. Improved RMPE (Regional Multi Person Estimator) : estimator
2. PF - Builder (Pose Flow Building) According to below figure (2)
3. PF NMS (Pose Flow non maximum suppersion) : reducing redundant link from adjacent frame

<p align="center">
  <img src="https://miro.medium.com/max/2202/1*zxVDN6bZakXivtcXD7vfyA.png" width="80%">.
</p>

In (2) it calculates Intra-Frame Pose Distance based on [RMPE: Regional Multi-Person Pose Estimation].

$$P_i$$  is a pose in a frame, i indicates the number of instances in one frame. It assumes that pose has m joints in one pose with denoted as $${\langle k^1_i, c^1_i \rangle,...,\langle k^m_i, c^m_i \rangle}$$. $$k$$ represent the position of joints and $$c$$ is the score of prediction in $$i^{th}$$ frame with m different keypoints.

Distance is denoted as $$d_{pose}(P_i,P_j)$$ and assuming $$B_i$$ is standing for box of $$P_i$$
$$ K_{sim}(P_1,P_2|\sigma_1) =
\begin{cases}
  \sum_{n}tanh \frac{c_1^n}{\sigma_1} tanh \frac{c_2^n}{\sigma_1} & {p_{2}^{n}} \text{is within} {B(p_{1}^{n})} \\    
  0    & \text{otherwise}
\end{cases}
$$

Question: Does tracking algorithm is for more precise accuracy instead of lightweightning of model?

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

기본적으로 estimation과 tracking은 각 키워드의 정의를 이해하면 쉬운 용어입니다. 또한 결과의 관점에서 보면 두 가지는 의미가 다르지만 결과는 완전히 같을 수도 있습니다. 사람의 pose를 tracking하는 방법에는 두 가지가 있는데, 하나는 top-down이고 다른 하나는 bottom-up입니다. 제가 이해하기로는 estimation은 주로 이미지에서, tracking은 비디오에서 사용됩니다.

Top-down 방식은 먼저 box를 검출한 뒤, box 안에서 keypoint를 표시하고 선으로 연결하여 pose를 그리는 방식입니다. Bottom-up 방식은 먼저 모든 keypoint를 검출한 뒤 결과를 보여주는 방식입니다.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*DMdb6SwPEeQBvqbFF6bXNg.jpeg" width="40%">.
</p>

[Pose Flow: Efficient Online Pose Tracking from Shanghai Jiao Tong University]에 따르면, top-down 방식이 accuracy와 tracking 속도 모두에서 훨씬 효과적입니다. Accuracy의 정의에는 두 가지 평가 지표가 있는데, 각각 mAP(mean average precision)와 MOTA(multiple object tracking accuracy, precision이 아님)입니다. 

새로운 용어
1. Improved RMPE (Regional Multi Person Estimator) : estimator
2. PF - Builder (Pose Flow Building) 아래 그림 (2) 참고
3. PF NMS (Pose Flow non maximum suppersion) : 인접 frame 간의 중복된 link를 줄임

<p align="center">
  <img src="https://miro.medium.com/max/2202/1*zxVDN6bZakXivtcXD7vfyA.png" width="80%">.
</p>

(2)에서는 [RMPE: Regional Multi-Person Pose Estimation]을 기반으로 Intra-Frame Pose Distance를 계산합니다.

$$P_i$$ 는 한 frame 내의 pose이며, i는 한 frame 내 instance의 번호를 나타냅니다. 하나의 pose가 m개의 joint를 가진다고 가정하며, 이를 $${\langle k^1_i, c^1_i \rangle,...,\langle k^m_i, c^m_i \rangle}$$로 표기합니다. $$k$$는 joint의 위치를 나타내고, $$c$$는 m개의 서로 다른 keypoint를 가진 $$i^{th}$$ frame에서의 prediction score입니다.

Distance는 $$d_{pose}(P_i,P_j)$$로 표기하며, $$B_i$$는 $$P_i$$의 box를 나타낸다고 가정합니다.
$$ K_{sim}(P_1,P_2|\sigma_1) =
\begin{cases}
  \sum_{n}tanh \frac{c_1^n}{\sigma_1} tanh \frac{c_2^n}{\sigma_1} & {p_{2}^{n}} \text{is within} {B(p_{1}^{n})} \\    
  0    & \text{otherwise}
\end{cases}
$$

질문: tracking 알고리즘은 모델 경량화보다는 더 높은 accuracy를 위한 것일까요?

참고 문헌

[1] https://arxiv.org/pdf/1802.00977.pdf 

[2] https://arxiv.org/pdf/1712.09184.pdf 

[3] https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

[4] https://motchallenge.net/results/3D_MOT_2015/?chl=3&orderBy=MOTA&orderStyle=DESC&det=Public

[5] https://arxiv.org/pdf/1612.00137.pdf

GitHub

[1] https://github.com/YuliangXiu/PoseFlow

</div>
