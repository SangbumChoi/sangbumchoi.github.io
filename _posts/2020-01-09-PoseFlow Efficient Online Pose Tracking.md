---
title: 'PoseFlow: Efficient Online Pose Tracking'
date: 2020-01-09
permalink: /posts/2020/01/PoseFlow Efficient Online Pose Tracking/
tags:
  - Computer Vision
  - Object Tracking
---

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
