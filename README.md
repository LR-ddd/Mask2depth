## Mask2depth:Depth-Guided Boundary Learning for Agricultural Image Segmentation
这款模型依赖零样本深度估计模型Depth_Pro的深度权重信息**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**对以下几位作者致以崇高的敬意
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

我们的团队由以下几位同事共同完成
*Liao Bin,Zhou BaoPing,Li XiaoFei, Qiu GuoYing*
![](data/network.jpg)
We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.


The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

