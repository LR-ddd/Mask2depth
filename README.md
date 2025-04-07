## Mask2depth:Depth-Guided Boundary Learning for Agricultural Image Segmentation
这款模型依赖零样本深度估计模型Depth_Pro的深度权重信息**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**对以下几位作者致以崇高的敬意
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

我们的工作由以下几位同事共同完成
*Liao Bin,Zhou BaoPing,Li XiaoFei, Qiu GuoYing*.

![](datas/network.jpg)


Mask2Depth在三种对边界敏感的农业场景超取得了显著成效。边界定量分析显示，在5像素容差范围内边界F1分数相较于传统方法提升2-5倍。可视化验证证实其对叶片断裂、土壤伪影等干扰具有卓越鲁棒性。本研究推动了边界敏感的农业视觉系统发展，为精细定位的精准农业应用提供新范式。我们的数据同样做了开源，数据链接为**[https://www.scidb.cn/detail?dataSetId=0dadce6de3c44354bcd73e09e7699410]**.

![](datas/vision.png).

## 开始安装
我们建议使用conda虚拟环境，“Mask2Dpeth”的安装命令如下

```bash
conda create -n mask2depth -y python=3.10
conda activate mask2depth

pip install -e .
```
深度信息权重下载
```bash
sh get_pretrained_models.sh
```

## 执行训练
首先需要修改src/mask2depth/cfg/config.yaml文件，请务必按照里面写的用法修改训练必须配置
**执行如下命令获取COCO格式标注文件的类别权重**
```bash
cd src/mask2depth

python utils.py
```

