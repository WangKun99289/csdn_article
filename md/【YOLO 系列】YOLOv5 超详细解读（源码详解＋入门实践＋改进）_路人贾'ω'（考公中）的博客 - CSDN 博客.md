> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129312409)

#### ![](https://img-blog.csdnimg.cn/36f240ea87334d0084f417bd20083ac5.gif)

![](https://img-blog.csdnimg.cn/d6a153ab12e54987b4c41df440b1a382.gif)
---------------------------------------------------------------------

前言
--

吼吼！终于来到了 YOLOv5 啦！

首先，一个热知识：YOLOv5 没有发表正式论文哦~

为什么呢？可能 YOLOv5 项目的作者 Glenn Jocher 还在吃帽子吧，hh

![](https://img-blog.csdnimg.cn/087a5c5747cd4a0e939c04587f78ba97.png)

**目录**
------

[前言](#%E5%89%8D%E8%A8%80)

[一、YOLOv5 的网络结构](#t0)

 [二、输入端](#t1)

[（1）Mosaic 数据增强](#t2)

[（2）自适应锚框计算](#t3)

[（3）自适应图片缩放](#t4)

[三、Backbone](#t5)

[（1）Focus 结构](#t6)

[（2）CSP 结构](#t7)

[四、Neck](#t8)

[五、Head](#t9)

[（1）Bounding box 损失函数](#t10)

[（2）NMS 非极大值抑制](#t11)

 [六、训练策略](#t12)

![](https://img-blog.csdnimg.cn/9c99667b6cde413fbb8d7d55d2a8a5fc.gif)

**【写论文必看】**[深度学习纯小白如何从零开始写第一篇论文？看完这篇豁然开朗！-CSDN 博客](https://blog.csdn.net/weixin_43334693/article/details/133617849?spm=1001.2014.3001.5501 "深度学习纯小白如何从零开始写第一篇论文？看完这篇豁然开朗！-CSDN博客")

**前期回顾：**

[【YOLO 系列】YOLOv4 论文超详细解读 2（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129248856?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv4论文超详细解读2（网络详解）")  
[【YOLO 系列】YOLOv4 论文超详细解读 1（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129232468?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv4论文超详细解读1（翻译 ＋学习笔记）")

[​​​​​​【YOLO 系列】YOLOv3 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129143961?spm=1001.2014.3001.5501 "​​​​​​【YOLO系列】YOLOv3论文超详细解读（翻译 ＋学习笔记）")  
[【YOLO 系列】YOLOv2 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129087464?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv2论文超详细解读（翻译 ＋学习笔记）")  
[【YOLO 系列】YOLOv1 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129011644?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv1论文超详细解读（翻译 ＋学习笔记）")

![](https://img-blog.csdnimg.cn/7a0e49a1df284620bf33616e1f85da2f.gif)🍀**本人 YOLOv5 源码详解系列：**

 [YOLOv5 源码逐行超详细注释与解读（1）——项目目录结构解析](https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析")  
[YOLOv5 源码逐行超详细注释与解读（2）——推理部分 detect.py](https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py")

[YOLOv5 源码逐行超详细注释与解读（3）——训练部分 train.py](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py")

[YOLOv5 源码逐行超详细注释与解读（4）——验证部分 val（test）.py](https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py")

[YOLOv5 源码逐行超详细注释与解读（5）——配置文件 yolov5s.yaml](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml")

[YOLOv5 源码逐行超详细注释与解读（6）——网络结构（1）yolo.py](https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py")

[YOLOv5 源码逐行超详细注释与解读（7）——网络结构（2）common.py](https://blog.csdn.net/weixin_43334693/article/details/129854764 "YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py")

![](https://img-blog.csdnimg.cn/7a0e49a1df284620bf33616e1f85da2f.gif)🌟**本人 YOLOv5 入门实践系列：** 

[YOLOv5 入门实践（1）——手把手带你环境配置搭建](https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501 "YOLOv5入门实践（1）——手把手带你环境配置搭建")

[YOLOv5 入门实践（2）——手把手教你利用 labelimg 标注数据集](https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501 "YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集")

[YOLOv5 入门实践（3）——手把手教你划分自己的数据集](https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501 "YOLOv5入门实践（3）——手把手教你划分自己的数据集")

[YOLOv5 入门实践（4）——手把手教你训练自己的数据集](https://blog.csdn.net/weixin_43334693/article/details/130043351?spm=1001.2014.3001.5501 "YOLOv5入门实践（4）——手把手教你训练自己的数据集")

[YOLOv5 入门实践（5）——从零开始，手把手教你训练自己的目标检测模型（包含 pyqt5 界面）](https://blog.csdn.net/weixin_43334693/article/details/130044342?spm=1001.2014.3001.5501 "YOLOv5入门实践（5）——从零开始，手把手教你训练自己的目标检测模型（包含pyqt5界面）")

  ![](https://img-blog.csdnimg.cn/7a0e49a1df284620bf33616e1f85da2f.gif)🌟**本人 YOLOv5 改进系列：** 

[YOLOv5 改进系列（0）——重要性能指标与训练结果评价及分析](https://blog.csdn.net/weixin_43334693/article/details/130564848?spm=1001.2014.3001.5501 "YOLOv5改进系列（0）——重要性能指标与训练结果评价及分析")

[YOLOv5 改进系列（1）——添加 SE 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/130551913?spm=1001.2014.3001.5501 "YOLOv5改进系列（1）——添加SE注意力机制")

[YOLOv5 改进系列（2）——添加 CBAM 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/130587102?spm=1001.2014.3001.5501 "YOLOv5改进系列（2）——添加CBAM注意力机制")

[YOLOv5 改进系列（3）——添加 CA 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/130619604?spm=1001.2014.3001.5501 "YOLOv5改进系列（3）——添加CA注意力机制")

[YOLOv5 改进系列（4）——添加 ECA 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/130641318?spm=1001.2014.3001.5501 "YOLOv5改进系列（4）——添加ECA注意力机制")

[YOLOv5 改进系列（5）——替换主干网络之 MobileNetV3](https://blog.csdn.net/weixin_43334693/article/details/130832933?spm=1001.2014.3001.5501 "YOLOv5改进系列（5）——替换主干网络之 MobileNetV3")

[YOLOv5 改进系列（6）——替换主干网络之 ShuffleNetV2](https://blog.csdn.net/weixin_43334693/article/details/131008642?spm=1001.2014.3001.5501 "YOLOv5改进系列（6）——替换主干网络之 ShuffleNetV2")

[YOLOv5 改进系列（7）——添加 SimAM 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/131031541?spm=1001.2014.3001.5501 "YOLOv5改进系列（7）——添加SimAM注意力机制")

[YOLOv5 改进系列（8）——添加 SOCA 注意力机制](https://blog.csdn.net/weixin_43334693/article/details/131053284?spm=1001.2014.3001.5501 "YOLOv5改进系列（8）——添加SOCA注意力机制")

[YOLOv5 改进系列（9）——替换主干网络之 EfficientNetv2](https://blog.csdn.net/weixin_43334693/article/details/131207097?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22131207097%22%2C%22source%22%3A%22weixin_43334693%22%7D "YOLOv5改进系列（9）——替换主干网络之EfficientNetv2")

[​​​​​​YOLOv5 改进系列（10）——替换主干网络之 GhostNet](https://blog.csdn.net/weixin_43334693/article/details/131235113?spm=1001.2014.3001.5501 "​​​​​​YOLOv5改进系列（10）——替换主干网络之GhostNet")

[YOLOv5 改进系列（11）——添加损失函数之 EIoU、AlphaIoU、SIoU、WIoU](https://blog.csdn.net/weixin_43334693/article/details/131350224?spm=1001.2014.3001.5501 "YOLOv5改进系列（11）——添加损失函数之EIoU、AlphaIoU、SIoU、WIoU")

[YOLOv5 改进系列（12）——更换 Neck 之 BiFPN](https://blog.csdn.net/weixin_43334693/article/details/131461294?spm=1001.2014.3001.5501 "YOLOv5改进系列（12）——更换Neck之BiFPN")

[YOLOv5 改进系列（13）——更换激活函数之 SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC 系列等](https://blog.csdn.net/weixin_43334693/article/details/131513850?spm=1001.2014.3001.5501 "YOLOv5改进系列（13）——更换激活函数之SiLU，ReLU，ELU，Hardswish，Mish，Softplus，AconC系列等")

[YOLOv5 改进系列（14）——更换 NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS](https://blog.csdn.net/weixin_43334693/article/details/131552028?spm=1001.2014.3001.5501 "YOLOv5改进系列（14）——更换NMS（非极大抑制）之 DIoU-NMS、CIoU-NMS、EIoU-NMS、GIoU-NMS 、SIoU-NMS、Soft-NMS")

[YOLOv5 改进系列（15）——增加小目标检测层](https://blog.csdn.net/weixin_43334693/article/details/131613721?spm=1001.2014.3001.5501 "YOLOv5改进系列（15）——增加小目标检测层")

[YOLOv5 改进系列（16）——添加 EMA 注意力机制（ICASSP2023 | 实测涨点）](https://blog.csdn.net/weixin_43334693/article/details/131973273?spm=1001.2014.3001.5501 "YOLOv5改进系列（16）——添加EMA注意力机制（ICASSP2023|实测涨点）")  
 [YOLOv5 改进系列（17）——更换 IoU 之 MPDIoU（ELSEVIER 2023 | 超越 WIoU、EIoU 等 | 实测涨点）](https://blog.csdn.net/weixin_43334693/article/details/131999141?spm=1001.2014.3001.5501 "YOLOv5改进系列（17）——更换IoU之MPDIoU（ELSEVIER 2023|超越WIoU、EIoU等|实测涨点）")

[YOLOv5 改进系列（18）——更换 Neck 之 AFPN（全新渐进特征金字塔 | 超越 PAFPN | 实测涨点）](https://blog.csdn.net/weixin_43334693/article/details/132070079?spm=1001.2014.3001.5501 "YOLOv5改进系列（18）——更换Neck之AFPN（全新渐进特征金字塔|超越PAFPN|实测涨点）")

[YOLOv5 改进系列（19）——替换主干网络之 Swin TransformerV1（参数量更小的 ViT 模型）](https://blog.csdn.net/weixin_43334693/article/details/132161488?spm=1001.2014.3001.5501 "YOLOv5改进系列（19）——替换主干网络之Swin TransformerV1（参数量更小的ViT模型）")

[YOLOv5 改进系列（20）——添加 BiFormer 注意力机制（CVPR2023 | 小目标涨点神器）](https://blog.csdn.net/weixin_43334693/article/details/132203200?spm=1001.2014.3001.5501 "YOLOv5改进系列（20）——添加BiFormer注意力机制（CVPR2023|小目标涨点神器）")

[YOLOv5 改进系列（21）——替换主干网络之 RepViT（清华 ICCV 2023 | 最新开源移动端 ViT）](https://blog.csdn.net/weixin_43334693/article/details/132211831?spm=1001.2014.3001.5501 "YOLOv5改进系列（21）——替换主干网络之RepViT（清华 ICCV 2023|最新开源移动端ViT）")

[YOLOv5 改进系列（22）——替换主干网络之 MobileViTv1（一种轻量级的、通用的移动设备 ViT）](https://blog.csdn.net/weixin_43334693/article/details/132367429 "YOLOv5改进系列（22）——替换主干网络之MobileViTv1（一种轻量级的、通用的移动设备 ViT）")

[YOLOv5 改进系列（23）——替换主干网络之 MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）](https://blog.csdn.net/weixin_43334693/article/details/132428203?spm=1001.2014.3001.5502 "YOLOv5改进系列（23）——替换主干网络之MobileViTv2（移动视觉 Transformer 的高效可分离自注意力机制）")  
[YOLOv5 改进系列（24）——替换主干网络之 MobileViTv3（移动端轻量化网络的进一步升级）](https://blog.csdn.net/weixin_43334693/article/details/133199471?spm=1001.2014.3001.5502 "YOLOv5改进系列（24）——替换主干网络之MobileViTv3（移动端轻量化网络的进一步升级）")

持续更新中。。。 

![](https://img-blog.csdnimg.cn/ff177a78130f4c159ba3ad33bbf7a56e.gif)  

**一、YOLOv5 的网络结构**
------------------

**YOLOv5 特点：** 合适于移动端部署，模型小，速度快

YOLOv5 有 **YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x** 四个版本。文件中，这几个模型的结构基本一样，不同的是 depth_multiple 模型深度和 width_multiple 模型宽度这两个参数。 就和我们买衣服的尺码大小排序一样，YOLOv5s 网络是 YOLOv5 系列中深度最小，特征图的宽度最小的网络。其他的三种都是在此基础上不断加深，不断加宽。

![](https://img-blog.csdnimg.cn/bd0fd3aafa384362a296ff41e4ee8a65.png)

**YOLOv5s 的网络结构如下：**

**![](https://img-blog.csdnimg.cn/e52b37d357b7467ca0c4edeaa8e6899d.png)**

**（1）输入端 ：** [Mosaic](https://so.csdn.net/so/search?q=Mosaic&spm=1001.2101.3001.7020) 数据增强、自适应锚框计算、自适应图片缩放

**（2）Backbone ：** Focus 结构，CSP 结构

**（3）Neck ：** FPN+PAN 结构

**（4）Head ：** CIOU_Loss

**基本组件：**

*   **Focus：**基本上就是 YOLO v2 的 passthrough。
*   **CBL：**由 Conv+Bn+Leaky_relu 激活函数三者组成。
*   **CSP1_X：**借鉴 CSPNet 网络结构，由三个卷积层和 X 个 Res unint 模块 Concate 组成。
*   **CSP2_X：**不再用 Res unint 模块，而是改为 CBL。
*   **SPP：**采用 1×1，5×5，9×9，13×13 的最大池化的方式，进行多尺度融合。

**YOLO5 算法性能测试图：**

![](https://img-blog.csdnimg.cn/c60f28642fda48128c57ea895c89271a.png)

 **二、输入端**
----------

### （1）Mosaic 数据增强

YOLOv5 在输入端采用了 Mosaic 数据增强，**Mosaic 数据增强算法将多张图片按照一定比例组合成一张图片，使模型在更小的范围内识别目标。**Mosaic [数据增强](https://so.csdn.net/so/search?q=%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA&spm=1001.2101.3001.7020)算法参考 CutMix 数据增强算法。CutMix 数据增强算法使用两张图片进行拼接，而 Mosaic 数据增强算法一般使用四张进行拼接，但两者的算法原理是非常相似的。

![](https://img-blog.csdnimg.cn/bc592324fcc44d56a3d88b567d3f7281.png)

**Mosaic 数据增强的主要步骤为：**

（1）随机选取图片拼接基准点坐标（xc，yc），另随机选取四张图片。

（2）四张图片根据基准点，分别经过尺寸调整和比例缩放后，放置在指定尺寸的大图的左上，右上，左下，右下位置。

（3）根据每张图片的尺寸变换方式，将映射关系对应到图片标签上。

（4）依据指定的横纵坐标，对大图进行拼接。处理超过边界的检测框坐标。

**采用 Mosaic 数据增强的方式有几个优点：**

**（1）丰富数据集：** 随机使用 4 张图像，随机缩放后随机拼接，增加很多小目标，大大增加了数据多样性。

**（2）增强模型鲁棒性：** 混合四张具有不同语义信息的图片，可以让模型检测超出常规语境的目标。

**（3）加强批归一化层（Batch Normalization）的效果：** 当模型设置 BN 操作后，训练时会尽可能增大批样本总量（BatchSize），因为 BN 原理为计算每一个特征层的均值和方差，如果批样本总量越大，那么 BN 计算的均值和方差就越接近于整个数据集的均值和方差，效果越好。

**（4）Mosaic 数据增强算法有利于提升小目标检测性能：** Mosaic 数据增强图像由四张原始图像拼接而成，这样每张图像会有更大概率包含小目标，从而提升了模型的检测能力。

###  （2）自适应锚框计算

**之前我们学的 YOLOv3、YOLOv4，对于不同的数据集，都会计算先验框 anchor。**然后在训练时，网络会在 anchor 的基础上进行预测，输出预测框，再和标签框进行对比，最后就进行梯度的反向传播。

在 YOLOv3、YOLOv4 中，训练不同的数据集时，是**使用单独的脚本进行初始锚框的计算**，在 YOLOv5 中，则是将此功能嵌入到整个训练代码里中。所以在每次训练开始之前，它都会根据不同的数据集来自适应计算 anchor。

but，如果觉得计算的[锚框](https://so.csdn.net/so/search?q=%E9%94%9A%E6%A1%86&spm=1001.2101.3001.7020)效果并不好，那你也可以在代码中将此功能关闭哈~

**自适应的计算具体过程：**

    ①获取数据集中所有目标的宽和高。

    ②将每张图片中按照等比例缩放的方式到 resize 指定大小，这里保证宽高中的最大值符合指定大小。

    ③将 bboxes 从相对坐标改成绝对坐标，这里乘以的是缩放后的宽高。

    ④筛选 bboxes，保留宽高都大于等于两个像素的 bboxes。

    ⑤使用 k-means 聚类三方得到 n 个 anchors，与 YOLOv3、YOLOv4 操作一样。

    ⑥使用遗传算法随机对 anchors 的宽高进行变异。倘若变异后的效果好，就将变异后的结果赋值给 anchors；如果变异后效果变差就跳过，默认变异 1000 次。这里是使用 anchor_fitness 方法计算得到的适应度 fitness，然后再进行评估。 

### （3）自适应图片缩放

**步骤：**

**(1) 根据原始图片大小以及输入到网络的图片大小计算缩放比例**

![](https://img-blog.csdnimg.cn/4ee7f39e1ced44c387520cc374e0c488.png)

原始缩放尺寸是 416*416，都除以原始图像的尺寸后，可以得到 0.52，和 0.69 两个缩放系数，选择小的缩放系数。

**(2) 根据原始图片大小与缩放比例计算缩放后的图片大小**

**![](https://img-blog.csdnimg.cn/98184dae0c0a44a68aea9f388e8d3c4f.png)**

原始图片的长宽都乘以最小的缩放系数 0.52，宽变成了 416，而高变成了 312。

**(3) 计算黑边填充数值**

**![](https://img-blog.csdnimg.cn/db1b5d9ea2134466b1a3971d7c926cc1.png)**

将 416-312=104，得到原本需要填充的高度。再采用 numpy 中 np.mod 取余数的方式，得到 8 个像素，再除以 2，即得到图片高度两端需要填充的数值。

**注意：**

（1）Yolov5 中填充的是**灰色**，即（114,114,114）。

（2）训练时没有采用缩减黑边的方式，还是采用传统填充的方式，即缩放到 416*416 大小。只是在测试，使用模型推理时，才采用缩减黑边的方式，提高目标检测，推理的速度。

（3）为什么 np.mod 函数的后面用 32？

因为 YOLOv5 的网络经过 5 次下采样，而 2 的 5 次方，等于 32。所以至少要去掉 32 的倍数，再进行取余。以免产生尺度太小走不完 stride（filter 在原图上扫描时，需要跳跃的格数）的问题，再进行取余。

**三、Backbone**
--------------

### （1）Focus 结构

**Focus 模块**在 YOLOv5 中是图片进入 **Backbone** 前，对图片进行切片操作，具体操作是在一张图片中每隔一个像素拿到一个值，类似于邻近下采样，这样就拿到了四张图片，四张图片互补，长得差不多，但是没有信息丢失，这样一来，将 W、H 信息就集中到了通道空间，输入通道扩充了 4 倍，**即拼接起来的图片相对于原先的 RGB 三通道模式变成了 12 个通道，最后将得到的新图片再经过卷积操作，最终得到了没有信息丢失情况下的二倍下采样特征图。**

以 YOLOv5s 为例，原始的 **640 × 640 × 3** 的图像输入 Focus 结构，采用切片操作，先变成 **320 × 320 × 12** 的特征图，再经过一次卷积操作，最终变成 **320 × 320 × 32** 的特征图。

切片操作如下：

![](https://img-blog.csdnimg.cn/bd574943892348d2828718dcd3d3ace3.png)

**作用：** 可以使信息不丢失的情况下提高计算力

**不足：**Focus 对某些设备不支持且不友好，开销很大，另外切片对不齐的话模型就崩了。

**后期改进：** **在新版中，YOLOv5 将 Focus 模块替换成了一个 6 x 6 的卷积层。**两者的计算量是等价的，但是对于一些 GPU 设备，使用 6 x 6 的卷积会更加高效。

![](https://img-blog.csdnimg.cn/15dac4d2963a4902868e74963f222b27.png)

###  （2）CSP 结构

YOLOv4 网络结构中，借鉴了 CSPNet 的设计思路，在主干网络中设计了 CSP 结构。

![](https://img-blog.csdnimg.cn/7c63d1f683dd4449908954253a515ce4.png)

YOLOv5 与 YOLOv4 不同点在于，**YOLOv4 中只有主干网络使用了 CSP 结构**。 而 **YOLOv5 中设计了两种 CSP 结构，以 YOLOv5s 网络为例，CSP1_ X 结构应用于 Backbone 主干网络，另一种 CSP2_X 结构则应用于 Neck 中。**

**![](https://img-blog.csdnimg.cn/1920fe563896490a9e1f10ec515c4b9d.png)**

**四、Neck**
----------

YOLOv5 现在的 Neck 和 YOLOv4 中一样，都采用 **FPN+PAN** 的结构。但是在它的基础上做了一些改进操作：**YOLOV4 的 Neck 结构中，采用的都是普通的卷积操作**，而 YOLOV5 的 Neck 中，采用 **CSPNet 设计的 CSP2 结构**，从而加强了网络特征融合能力。

结构如下图所示，FPN 层自顶向下传达强语义特征，而 PAN 塔自底向上传达定位特征：

![](https://img-blog.csdnimg.cn/0a1c60857cdf414f8a043ef07f1d3236.png)

**五、Head**
----------

### （1）Bounding box 损失函数

YOLO v5 采用 CIOU_LOSS 作为 bounding box 的损失函数。（关于 IOU_ Loss、GIOU_ Loss、DIOU_ Loss 以及 CIOU_Loss 的介绍，请看 YOLOv4 那一篇：[【YOLO 系列】YOLOv4 论文超详细解读 2（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129248856?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv4论文超详细解读2（网络详解）")）

### （2）NMS 非极大值抑制

NMS 的本质是搜索局部极大值，抑制非极大值元素。

非极大值抑制，**主要就是用来抑制检测时冗余的框**。因为在目标检测中，在同一目标的位置上会产生大量的候选框，这些候选框相互之间可能会有重叠，所以我们需要利用非极大值抑制找到最佳的目标边界框，消除冗余的边界框。

**算法流程：**

  1. 对所有预测框的置信度降序排序

  2. 选出置信度最高的预测框，确认其为正确预测，并计算他与其他预测框的 IOU

  3. 根据步骤 2 中计算的 IOU 去除重叠度高的，IOU > threshold 阈值就直接删除

  4. 剩下的预测框返回第 1 步，直到没有剩下的为止

 **SoftNMS：**

**当两个目标靠的非常近时，置信度低的会被置信度高的框所抑制**，那么当两个目标靠的十分近的时候就只会识别出一个 BBox。为了解决这个问题，可以使用 softNMS。

它的基本思想是用稍低一点的分数来代替原有的分数，而不是像 NMS 一样直接置零。

![](https://img-blog.csdnimg.cn/e1ac306ffa94418aba7048d321e7b1fe.png)

 **六、训练策略**
-----------

**（1）多尺度训练（Multi-scale training）。** 如果网络的输入是 416 x 416。那么训练的时候就会从 0.5 x 416 到 1.5 x 416 中任意取值，但所取的值都是 32 的整数倍。

**（2）训练开始前使用 warmup 进行训练。** 在模型预训练阶段，先使用较小的学习率训练一些 epochs 或者 steps (如 4 个 epoch 或 10000 个 step)，再修改为预先设置的学习率进行训练。

**（3）使用了 cosine 学习率下降策略（Cosine LR scheduler）。**

**（4）采用了 EMA 更新权重 (Exponential Moving Average)。** 相当于训练时给参数赋予一个动量，这样更新起来就会更加平滑。

**（5）使用了 amp 进行混合精度训练（Mixed precision）。** 能够减少显存的占用并且加快训练速度，但是需要 GPU 支持。

总结一下，YOLO v5 和前 YOLO 系列相比的改进：

*   (1) 增加了正样本：方法是邻域的正样本 anchor 匹配策略。
*   (2) 通过灵活的配置参数，可以得到不同复杂度的模型
*   (3) 通过一些内置的超参优化策略，提升整体性能
*   (4) 和 yolov4 一样，都用了 mosaic 增强，提升小物体检测性能

![](https://img-blog.csdnimg.cn/b9eeb282d47b4b98b2620d7b3f912169.gif)