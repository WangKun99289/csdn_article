> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129356033)

#### ![](https://img-blog.csdnimg.cn/2872fc314a4242c99cbfdcdc2aec89c2.gif)

![](https://img-blog.csdnimg.cn/810179b6b85a4b188ab4ca83c87d9f5d.jpeg)

前言
--

前面简单介绍了 YOLOv5 的网络结构和创新点（直通车：[【YOLO 系列】YOLOv5 超详细解读（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129312409?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv5超详细解读（网络详解）")）

在接下来我们会进入到 YOLOv5 更深一步的学习，首先从源码解读开始。

因为我是纯小白，刚开始下载完源码时真的一脸懵，所以就先从最基础的**项目目录结构**开始吧~ 因为相关解读不是很多，所以有的是我根据作者给的英文文档自己翻译的，如有不对之处欢迎大家指正呀！这篇只是简单介绍每个文件是做什么的，大体上了解这个项目，具体的代码详解后期会慢慢更新，也欢迎大家关注我的专栏，和我一起学习呀！

源码下载地址：[mirrors / ultralytics / yolov5 · GitCode](https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator "mirrors / ultralytics / yolov5 · GitCode")

![](https://img-blog.csdnimg.cn/0660a36fc18b4cfbaece95774eb62c6b.gif)

![](https://img-blog.csdnimg.cn/09ee3e185d0f45bbb505b24f1e9adb4d.gif)**【写论文必看】**[深度学习纯小白如何从零开始写第一篇论文？看完这篇豁然开朗！](https://blog.csdn.net/weixin_43334693/article/details/133617849?spm=1001.2014.3001.5501 "深度学习纯小白如何从零开始写第一篇论文？看完这篇豁然开朗！") 

![](https://img-blog.csdnimg.cn/09ee3e185d0f45bbb505b24f1e9adb4d.gif)**🍀本人 YOLOv5 源码详解系列：** 

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

目录
--

[前言](#%E5%89%8D%E8%A8%80)

 [一、项目目录结构](#%C2%A0%E4%B8%80%E3%80%81%E9%A1%B9%E7%9B%AE%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84)

 [1.1 .github 文件夹](#t1)

 [1.2 datasets](#t2)

 [1.3 data 文件夹](#t3)

 [1.4 models 文件夹](#t4)

 [1.5 runs 文件夹](#t5)

 [1.6 utils 文件夹](#t6)

 [1.7 其他一级目录文件](#t7)

![](https://img-blog.csdnimg.cn/a59d18afca644dbaadcabe7a4bd428ff.gif)
---------------------------------------------------------------------

 一、项目[目录结构](https://so.csdn.net/so/search?q=%E7%9B%AE%E5%BD%95%E7%BB%93%E6%9E%84&spm=1001.2101.3001.7020)
---------------------------------------------------------------------------------------------------------

![](https://img-blog.csdnimg.cn/fe36b44ff07e4cd6acc2ce672b11b657.png)​

将源码下载好并配置好环境之后，就可以看到 YOLOv5 的整体目录如上图所示。

接下来我们逐一分析

###  1.1 .github 文件夹

![](https://img-blog.csdnimg.cn/d81b6ebbfcef4e458f299020f26072c1.png)​

 **github** 是存放关于 github 上的一些 “配置” 的，这个不重要，我们可以不管它。

### 1.2 datasets

![](https://img-blog.csdnimg.cn/6ec34d576a784ddbbbf51571337a9881.png)​

我们刚下载下来的源码是不包含这个文件夹的，**datasets 用来存放自己的数据集，分为 images 和 labels 两部分**。**同时每一个文件夹下，又应该分为 train，val。**.cache 文件为缓存文件，将数据加载到内存中，方便下次调用快速。可以自命名，比如我的火焰数据集就叫 “fire_yolo_format”。

###  1.3 data 文件夹

 ![](https://img-blog.csdnimg.cn/d9a15bc9e42c4162b9229f02ab4eb251.png)​

**data 文件夹**主要是存放一些**超参数的配置文件**（如. yaml 文件）是用来配置训练集和测试集还有验证集的路径的，其中还包括目标检测的种类数和种类的名称；还有**一些官方提供测试的图片**。YOLOv5 有大约 30 个超参数用于各种训练设置。更好的初始猜测会产生更好的最终结果，因此在演化之前正确初始化这些值很重要。

如果是**训练自己的数据集的话，那么就需要修改其中的 yaml 文件**。不过要注意，自己的数据集不建议放在这个路径下面，建议把数据集放到 YOLOv5 项目的同级目录下面。

**详解：**

*   **hyps 文件夹**   # 存放 yaml 格式的超参数配置文件
    *   **hyps.scratch-high.yaml** # 数据增强高，适用于大型型号，即 v3、v3-spp、v5l、v5x
        
    *   **hyps.scratch-low.yaml**  # 数据增强低，适用于较小型号，即 v5n、v5s
        
    *   **hyps.scratch-med.yaml**  # 数据增强中，适用于中型型号。即 v5m
        
*   **images**  # 存放着官方给的两张测试图片
*   **scripts**  # 存放数据集和权重下载 shell 脚本
    *   **download_weights.sh**  # 下载权重文件，包括五种大小的 P5 版和 P6 版以及分类器版
    *   **get_coco.sh**   # 下载 coco 数据集
        
    *   **get_coco128.sh** # 下载 coco128（只有 128 张）
        
*   **Argoverse.yaml**  # 后面的每个. yaml 文件都对应一种标准数据集格式的数据
    
*   **coco.yaml**   # COCO 数据集配置文件
*   **coco128.yaml**   # COCO128 数据集配置文件
*   **voc.yaml**   # VOC 数据集配置文件

###  1.4 models 文件夹

![](https://img-blog.csdnimg.cn/bb7bb2f59ed94ca38d1c3039451e9374.png)​

**models** 是模型文件夹。里面主要是一些网络构建的配置文件和函数，其中包含了该项目的四个不同的版本，分别为是 **s、m、l、x**。从名字就可以看出，这几个版本的大小。**他们的检测速度分别都是从快到慢，但是精确度分别是从低到高。**如果训练自己的数据集的话，就需要修改这里面相对应的 yaml 文件来训练自己模型。

**详解：**

*   **hub**  # 存放 yolov5 各版本目标检测网络模型配置文件
    *   **anchors.yaml**  # COCO 数据的默认锚点
    *   **yolov3-spp.yaml**  # 带 spp 的 yolov3
    *   **yolov3-tiny.yaml**  # 精简版 yolov3
    *   **yolov3.yaml**  # yolov3
    *   **yolov5-bifpn.yaml ** # 带二值 fpn 的 yolov5l
    *   **yolov5-fpn.yaml**  # 带 fpn 的 yolov5
    *   **yolov5-p2.yaml**  # (P2, P3, P4, P5) 都输出，宽深与 large 版本相同，相当于比 large 版本能检测更小物体
    *   **yolov5-p34.yaml ** # 只输出 (P3, P4)，宽深与 small 版本相同，相当于比 small 版本更专注于检测中小物体
    *   **yolov5-p6.yaml**  # (P3, P4, P5, P6) 都输出，宽深与 large 版本相同，相当于比 large 版本能检测更大物体
    *   **yolov5-p7.yaml**  # (P3, P4, P5, P6, P7) 都输出，宽深与 large 版本相同，相当于比 large 版本能检测更更大物体
    *   **yolov5-panet.yaml**  # 带 PANet 的 yolov5l
    *   **yolov5n6.yaml ** # (P3, P4, P5, P6) 都输出，宽深与 nano 版本相同，相当于比 nano 版本能检测更大物体，anchor 已预定义
    *   **yolov5s6.yaml**  # (P3, P4, P5, P6) 都输出，宽深与 small 版本相同，相当于比 small 版本能检测更大物体，anchor 已预定义
    *   **yolov5m6.yaml**   # (P3, P4, P5, P6) 都输出，宽深与 middle 版本相同，相当于比 middle 版本能检测更大物体，anchor 已预定义
    *   **yolov5l6.yaml  ** # (P3, P4, P5, P6) 都输出，宽深与 large 版本相同，相当于比 large 版本能检测更大物体，anchor 已预定义，推测是作者做实验的产物
    *   **yolov5x6.yaml  ** # (P3, P4, P5, P6) 都输出，宽深与 Xlarge 版本相同，相当于比 Xlarge 版本能检测更大物体，anchor 已预定义
    *   **yolov5s-ghost.yaml**   # backbone 的卷积换成了 GhostNet 形式的 yolov5s，anchor 已预定义
    *   **yolov5s-transformer.yaml**  # backbone 最后的 C3 卷积添加了 Transformer 模块的 yolov5s，anchor 已预定义
*   **_int_.py**   # 空的
*   **common.py**   # 放的是一些网络结构的定义通用模块，包括 autopad、Conv、DWConv、TransformerLayer 等
*   **experimental.py**   # 实验性质的代码，包括 MixConv2d、跨层权重 Sum 等
*   **tf.py**  # tensorflow 版的 yolov5 代码
*   **yolo.py**  # yolo 的特定模块，包括 BaseModel，DetectionModel，ClassificationModel，parse_model 等
*   **yolov5l.yaml**   # yolov5l 网络模型配置文件，large 版本，深度 1.0，宽度 1.0
*   **yolov5m.yaml**   # yolov5m 网络模型配置文件，middle 版本，深度 0.67，宽度 0.75
*   **yolov5n.yaml**   # yolov5n 网络模型配置文件，nano 版本，深度 0.33，宽度 0.25
*   **yolov5s.yaml**   # yolov5s 网络模型配置文件，small 版本，深度 0.33，宽度 0.50
*   **yolov5x.yaml**   # yolov5x 网络模型配置文件，Xlarge 版本，深度 1.33，宽度 1.25

### 1.5 runs 文件夹

![](https://img-blog.csdnimg.cn/a025322400924df59f6b6a99053e9cb1.png)

**runs** 是我们运行的时候的一些输出文件。每一次运行就会生成一个 exp 的文件夹。

![](https://img-blog.csdnimg.cn/a896c5a0bab946f19da379d7fb4c086c.png)

 **详解：**

*   **detect **  # 测试模型，输出图片并在图片中标注出物体和概率
*   **train**    # 训练模型，输出内容，模型 (最好、最新) 权重、混淆矩阵、F1 曲线、超参数文件、P 曲线、R 曲线、PR 曲线、结果文件（loss 值、P、R）等 expn  
     
    *   **expn**   # 第 n 次实验数据
    *   **confusion_matrix.png**   # 混淆矩阵
    *   **P_curve.png**   # 准确率与置信度的关系图线
    *   **R_curve.png**  # 精准率与置信度的关系图线
    *   **PR_curve.png**  #  精准率与召回率的关系图线
    *   **F1_curve.png**   # F1 分数与置信度（x 轴）之间的关系
    *   **labels_correlogram.jpg**   # 预测标签长宽和位置分布
    *    **results.png**   # 各种 loss 和 metrics（p、r、mAP 等，详见 utils/metrics）曲线
    *   **results.csv**  # 对应上面 png 的原始 result 数据
    *   **hyp.yaml**  #  超参数记录文件
    *   **opt.yaml**  # 模型可选项记录文件
    *   **train_batchx.jpg**  # 训练集图像 x（带标注）
    *   **val_batchx_labels.jpg**  # 验证集图像 x（带标注）
    *   **val_batchx_pred.jpg**  # 验证集图像 x（带预测标注）
    *   **weights**  #  权重
    *   **best.pt**  # 历史最好权重
    *   **last.pt**   # 上次检测点权重
    *   **labels.jpg**  # 4 张图， 4 张图，（1，1）表示每个类别的数据量

                                                               （1，2）真实标注的 bounding_box

                                                               （2，1） 真实标注的中心点坐标

                                                               （2，2）真实标注的矩阵宽高

### 1.6 utils 文件夹

   ![](https://img-blog.csdnimg.cn/14be36985e4f471194b2cc91759801d1.png)

 **utils** 工具文件夹。存放的是工具类的函数，里面有 loss 函数，metrics 函数，plots 函数等等。

 **详解：**

*   **aws**   #  恢复中断训练, 和 aws 平台使用相关的工具
*   **flask_rest_api**  # 和 flask 相关的工具
*   **google_app_engine**   # 和谷歌 app 引擎相关的工具
*   **loggers**    # 日志打印
*   **_init_.py** # notebook 的初始化，检查系统软件和硬件
*   **activations.py**  #  激活函数
*   **augmentations**  # 存放各种图像增强技术
*   **autoanchor.py**    #  自动生成锚框
*   **autobatch.py**   # 自动生成批量大小
*   **benchmarks.py**   #  对模型进行性能评估（推理速度和内存占用上的评估）
*   **callbacks.py**   #  回调函数，主要为 logger 服务
*   **datasets ** # dateset 和 dateloader 定义代码
*   **downloads.py**   #  谷歌云盘内容下载
*   **general.py**   # 全项目通用代码，相关实用函数实现
*   **loss.py**   #  存放各种损失函数
*   **metrics.py**   # 模型验证指标，包括 ap，混淆矩阵等
*   **plots.py**   #  绘图相关函数，如绘制 loss、ac 曲线，还能单独将一个 bbox 存储为图像
*   **torch_utils.py**   # 辅助函数

### 1.7 其他一级目录文件

![](https://img-blog.csdnimg.cn/4ed0ee4c60be4ce8a1cef106ff9deeb4.png)

 **详解：**

*   **.dockerignore**   # docker 的 ignore 文件
*   **.gitattributes**   # 用于将.[ipynb](https://so.csdn.net/so/search?q=ipynb&spm=1001.2101.3001.7020 "ipynb") 后缀的文件剔除 GitHub 语言统计
*   **.gitignore**   #  docker 的 ignore 文件
*   **CONTRIBUTING.md**  # markdown 格式说明文档
*   **detect.py**   # 目标检测预测脚本
*   **export.py**  #  模型导出
*   **hubconf.py**  # pytorch hub 相关
*   **LICENSE **   # 证书
*   **README.md **   # markdown 格式说明文档
*   **requirements.txt**  # 可以通过 pip install requirement 进行依赖环境下载
*   **setup.cfg**  #  项目打包文件
*   **train.py**   # 目标检测训练脚本
*   **tutorial.ipynb**  #  目标检测上手教程
*   **val.py**  # 目标检测验证脚本
*   **yolov5s.pt**   #  coco 数据集模型预训练权重，运行代码的时候会自动从网上下载

> 本文参考：
> 
> [YOLOV5 学习笔记（四）——项目目录及代码讲解](https://blog.csdn.net/HUASHUDEYANJING/article/details/126086708 "YOLOV5学习笔记（四）——项目目录及代码讲解")
> 
> [YOLOv5-6.2 版本代码 Project 逐文件详解](https://blog.csdn.net/qq_53627591/article/details/128555629 "YOLOv5-6.2版本代码Project逐文件详解")

![](https://img-blog.csdnimg.cn/307c53c22fd04be1896c127723aca252.gif)