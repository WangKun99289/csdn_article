> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129232468?spm=1001.2014.3001.5501)

![](https://img-blog.csdnimg.cn/ee845a4040d443b38d6fd00b02b1ccee.gif)

前言 
---

      经过上一期的开篇介绍，我们知道 [YOLO](https://so.csdn.net/so/search?q=YOLO&spm=1001.2101.3001.7020) 之父 Redmon 在 twitter 正式宣布退出 cv 界，大家都以为 YOLO 系列就此终结的时候，天空一声巨响，YOLOv4 闪亮登场！v4 作者是 AlexeyAB 大神，虽然换人了，但论文中给出的测试结果依然保留 YOLO 系列的血统：保持相对较高的 mAP 的同时，大大降低计算量，可谓是学术成果在工业应用的典范，至于实际使用如何，还需要时间的进一步检验。  
        [YOLOv4](https://so.csdn.net/so/search?q=YOLOv4&spm=1001.2101.3001.7020) 的论文是我读文献以来最不 “爽” 的一篇，YOLOv4 像一个“缝合怪”，几乎没有提出什么创新性的东西，其实是一个结合了大量前人研究技术，加以组合并进行适当创新的算法，实现了速度和精度的完美平衡。里面涉及的 tricks 过多，每读到一点我都要查大量资料。由于篇幅有限，本篇只是对论文进行解读，trick 详解请看这篇：[【YOLO 系列】YOLOv4 论文超详细解读 2（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129248856 "【YOLO系列】YOLOv4论文超详细解读2（网络详解）")

       好了，我们现在开始吧~

**学习资料：** 

论文链接：[《YOLOv4：Optimal Speed and Accuracy of Object Detection》](https://arxiv.org/pdf/2004.10934.pdf "《YOLOv4：Optimal Speed and Accuracy of Object Detection》")

代码链接：[mirrors / alexeyab / darknet · GitCode](https://gitcode.net/mirrors/alexeyab/darknet?utm_source=csdn_github_accelerator "mirrors / alexeyab / darknet · GitCode") 

**前期回顾：**  

[【YOLO 系列】YOLOv3 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129143961?spm=1001.2014.3001.5502 "【YOLO系列】YOLOv3论文超详细解读（翻译 ＋学习笔记）")  
[  
【YOLO 系列】YOLOv2 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129087464?spm=1001.2014.3001.5502 "【YOLO系列】YOLOv2论文超详细解读（翻译 ＋学习笔记）") 

[【YOLO 系列】YOLOv1 论文超详细解读（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129011644?spm=1001.2014.3001.5502 "【YOLO系列】YOLOv1论文超详细解读（翻译 ＋学习笔记）")

目录
--

[前言](#%E5%89%8D%E8%A8%80%C2%A0) 

 [Abstract—摘要](#t0)

 [一、 Introduction—简介](#t1)

[二、Related work—相关工作](#t2)

[2.1 Object detection models—目标检测模型](#t3)

[2.2 Bag of freebies](#t4)

[2.3 Bag of specials](#t5)

[三、Methodology—方法](#t6)

[3.1 Selection of architecture—架构选择](#t7) 

[3.2 Selection of BoF and BoS—BoF 和 BoS 的选择](#t8)

[3.3 Additional improvements—进一步改进](#t9)

[3.4 YOLOv4](#t10)

[四、Experiments—实验](#t11)

[4.1 Experimental setup—实验设置](#t12)

[4.2 Influence of different features on Classifier training—不同特征对分类器训练的影响](#t13)

[4.3 Influence of different features on Detector training—不同特征对检测器训练的影响](#t14)

[4.4 Influence of different backbones and pre- trained weightings on Detector training—不同的 backbone 和预先训练权重对检测器训练的影响](#t15)

[4.5 Influence of different mini-batch size on Detec- tor training—不同的小批尺寸对检测器培训的影响](#t16)

 [五、Results—结果](#t17)

 **Abstract—摘要**
----------------

### **翻译**

大量的特征据说可以提高卷积神经网络 (CNN) 的精度。需要在大数据集上对这些特征的组合进行实际测试，并对结果进行理论证明。有些特性只适用于某些模型，只适用于某些问题，或仅适用于小规模数据集；而一些特性，如批处理标准化和残差连接，适用于大多数模型、任务和数据集。我们假设这些普遍特征包括加权残差连接 (WRC)、跨阶段部分连接(CSP)、交叉小批归一化(CmBN)、自我对抗训练(SAT) 和 Mish 激活。我们使用新功能：WRC，CSP，CmBN，SAT，Mish 激活，Mosaic 数据增强、CmBN，DropBlock 正则化和 CIoU 损失，并结合其中一些实现最先进的结果：43.5%AP，(65.7%AP50)的实时速度∼65FPS Tesla V100。源代码是在 https://github.com/AlexeyAB/darknet.。 

### **精读**

#### **提高 CNN 准确性的方法**

（1）**专用特性：** 一些特征只针对某一模型，某一问题，或仅为小规模数据集

（2）**通用特性：** 一些特性，如批处理规范化和残差连接，则适用于大多数模型、任务和数据集。这些通用特性包括加权剩余连接 (WRC)、跨阶段部分连接(CSP)、跨小批标准化(CmBN)、自反训练(SAT) 和 Mish 激活函数。

#### YOLOv4 使用的技巧

**使用新特性：**WRC、CSP、CmBN、SAT、Mish 激活函数、Mosaic 数据增强、CmBN、DropBlock 正则化、CIoU 损失，结合这些技巧实现先进的结果。

#### 实现结果

在 Tesla V100 上，MS COCO 数据集以 65 FPS 的实时速度达到 43.5 % AP (65.7 % AP50)。

 **一、 Introduction—简介**
-----------------------

### **翻译**

大多数基于 cnn 的对象检测器基本上只适用于推荐系统。例如，通过城市摄像机搜索免费停车位是由慢速精确的模型执行的，而汽车碰撞警告与快速不准确的模型有关。为了提高实时目标检测器的精度，不仅可以将它们用于提示生成推荐系统，还可以用于独立的流程管理和减少人工输入。在传统图形处理单元 (GPU) 上的实时对象检测器操作允许它们以可承受的价格大规模使用。最精确的现代神经网络不能实时运行，需要大量的 gpu 来进行大的小批量训练。我们通过创建一个在普通的 GPU 上实时运行的 CNN 来解决这些问题，而其训练只需要一个普通的 GPU。 

        这项工作的主要目标是在生产系统中设计一个目标检测器的快速运行速度，并优化并行计算，而不是低计算体积理论指标 (BFLOP)。我们希望所设计的对象能够方便地训练和使用。例如，任何使用普通的 GPU 进行训练和测试的人都可以实现实时、高质量和令人信服的目标检测结果，如图 1 所示的 YOLOv4 结果所示。我们的贡献总结如下：

         1. 我们开发了一个高效而强大的目标检测模型。它使每个人都可以使用一个 1080 Ti 或 2080 Ti GPU 来训练一个超快和准确的目标探测器。

        2. 我们验证了 state-of-the-art Bag-of Freebies and Bag-of-Specials 对目标检测的影响。

         3. 我们修改了最先进的方法，使其更有效，更适合于单一的 GPU 训练，包括 CBN[89]，PAN[49]，SAM[85] 等。

### **精读**

#### 启发

（1）**改进性能：** 大多数基于 CNN 的目标检测器主要只适用于推荐系统，因此需要提高实时目标探测器的准确性。

（2）**单 GPU 训练：** 最精确的现代神经网络不能实时运行，需要大量的 GPU 来进行大规模的小批量训练。我们通过创建一个在常规 GPU 上实时运行的 CNN 来解决这些问题，而训练只需要一个常规 GPU。

#### 目的

设计生产系统中目标检测器的快速运行速度，优化并行计算，而不是低计算量理论指标 （BFLOP）。

#### 贡献

（1）开发了一个高效、强大的目标检测模型。使用单个 1080 Ti 或 2080 Ti GPU 就能训练一个超级快速和精确的目标探测器。

（2）验证了在检测器训练过程中，最先进的 Bag-of-Freebies 和 Bag-of-Specials 对目标检测方法的影响。

（3）修改了最先进的方法，使其更有效，更适合于单 GPU 训练。

> Q： Bag-of-Freebies 和 Bag-of-Specials
> 
> **Bag-of-Freebies：** 指不会显著影响模型测试速度和模型复杂度的技巧，主要就是数据增强操作、标签软化等外在训练方法，即不需要改变网络模型。
> 
> **Bag-of-Specials：** 是用最新最先进的方法（网络模块）来魔改检测模型。只增加少量推理成本但能显著提高对象检测精度的插件模块和后处理方法，一般来说，这些插件模块都是为了增强模型中的某些属性，如扩大感受野、引入注意力机制或加强特征整合能力等，而后处理是筛选模型预测结果的一种方法。

**二、Related work—相关工作**
-----------------------

### 2.1 Object detection models—目标检测模型

### **翻译**

现代探测器通常由两部分组成，一个是在 ImageNet 上预先训练的主干，另一个是用于预测物体的类和边界框的头部。对于那些运行在 GPU 平台上的检测器，它们的主干可以是 VGG[68]、ResNet[26]、ResNeXt[86]或 DenseNet[30]。对于那些运行在 CPU 平台上的检测器，它们的主干可以是 SqueezeNet [31]、MobileNet[28,66,27,74]或 ShufflfleNet[97,53]。对于头部部分，通常可分为一级目标探测器和两级目标探测器两类。最具代表性的两级目标探测器是 R-CNN[19]系列，包括 Fast R-CNN[18]、Faster R-CNN[64]、R-FCN[9]和 Libra R-CNN[58]. 也可以使一个两级目标检测器成为一个无锚点的目标检测器，如反应点 [87]。对于单级目标探测器，最具代表性的模型是 YOLO[61,62,63]、SSD[50] 和 RetinaNet[45]。近年来，无锚的单级目标探测器已经发展起来。这类检测器有 CenterNet [13]、CornerNet [37,38]、FCOS[78]等。近年来开发的目标探测器经常在主干和头部之间插入一些层，这些层通常用于收集不同阶段的特征图。我们可以称之为物体探测器的颈部。通常，颈部由几条自下向上的路径和几条自上向下的路径组成。配备这种机制的网络包括特征金字塔网络 (FPN)[44]、路径聚合网络(PAN)[49]、BiFPN[77] 和 NAS-FPN[17]。

        除了上述模型外，一些研究人员还强调了直接构建一个新的主干 (DetNet[43]，DetNAS[7]) 或一个新的整体模型 (SpineNet[12]，HitDetector[20]) 用于目标检测。 

        综上所述，一个普通的物体探测器由以下几个部分组成：  
![](https://img-blog.csdnimg.cn/4a5aca1c4f124f83b9cc134c7fab13c5.png)

### **精读**

#### 现代目标检测器组成

**（1）主干 backbone：** 在 ImageNet 上预先训练的网络用来特征提取。

*   在 **GPU** 平台上运行的检测器，主干可以是 VGG、ResNet、ResNeXt 或 DenseNet。
*   在 **CPU** 平台上运行的检测器，主干可以是 SqueezeNet、MobileNet 或 ShuffleNet。

**（2）头部 head：** 对图像特征进行预测，生成边界框和并预测类别。通常分为两类即单阶段目标检测器和两阶段目标检测器。

*   **two stage：** R-CNN 系列，包括 fast R-CNN、faster R-CNN、R-FCN 和 Libra R-CNN。
*   **one stage：** 最具代表性的模型有 YOLO、SSD 和 RetinaNet。

**（3）颈部 neck：** 近年来发展起来的目标检测器常常在主干和头部之间插入一系列混合和组合图像特征的网络层，并将图像特征传递到预测层。称之为目标检测器的颈部 neck。

通常，一个颈部 neck 由几个自下而上的路径和几个自上而下的路径组成。具有该机制的网络包括特征金字塔网络 (FPN)、路径汇聚网络 (PAN)、BiFPN 和 NAS-FPN。

综上所述，一个普通的物体探测器是由 “特征输入、骨干网络、颈部和头部” 四部分组成的：

![](https://img-blog.csdnimg.cn/69324c0f13164004aed869a95b92cfef.png)

### 2.2 Bag of freebies

### 翻译

通常，一个传统的目标检测器是离线训练的。因此，研究者总是喜欢利用这一优势，开发出更好的训练方法，使目标检测器在不增加推理成本的情况下获得更好的精度。我们把这些只会改变培训策略或只增加培训成本的方法称为 “bag of freebies”。目标检测方法经常采用的、满足 bag of freebies. 定义的是数据增强。数据增强的目的是为了增加输入图像的可变性，从而使所设计的目标检测模型对在不同环境下获得的图像具有更高的鲁棒性。例如，光度畸变和几何畸变是两种常用的数据增强方法，它们肯定有利于目标检测任务。在处理光度失真时，我们会调整图像的亮度、对比度、色调、饱和度和噪声。对于几何失真，我们添加了随机缩放、裁剪、翻转和旋转。 

        上述数据增强方法都是像素级调整，并保留调整区域中的所有原始像素信息。此外，一些从事数据增强工作的研究人员将其重点放在了模拟对象遮挡问题上。它们在图像分类和目标检测方面取得了良好的效果。例如，随机擦除 [100] 和 CutOut[11]可以随机选择图像中的矩形区域，并填充一个随机的或互补的零值。对于 hide-and-seek[69]和 grid mask[6]，它们随机或均匀地选择一个图像中的多个矩形区域，并将它们替换为所有的零。如果将类似的概念应用于特征映射，则会有 DropOut[71]、Drop 连接 [80] 和 DropBlock[16]方法。此外，一些研究者提出了使用多个图像一起进行数据增强的方法。例如，MixUp[92]使用两幅图像用不同的系数比进行乘法和叠加，然后用这些叠加的比率来调整标签。 

         CutMix[91] 是将裁剪后的图像覆盖到其他图像的矩形区域，并根据混合区域的大小调整标签。除上述方法外，还采用了样式转移 GAN[15] 进行数据增强，这种使用可以有效地减少 CNN 学习到的纹理偏差。

        与上面提出的各种方法不同，其他一些 bag of freebies 都致力于解决数据集中的语义分布可能存在偏差的问题。在处理语义分布偏差问题时，一个非常重要的问题是不同类之间存在数据不平衡的问题，这个问题通常通过两级对象检测器中的硬负例挖掘 [72] 或在线硬例挖掘 [67] 来解决。但该示例挖掘方法不适用于单级对象检测器，因为这种检测器属于密集预测体系结构。因此，Lin 等人 [45] 提出了焦点损失来解决不同类之间存在的数据不平衡问题。另一个非常重要的问题是，很难表达不同类别之间的关联程度与单一热硬表示之间的关系。这种表示方案经常用于执行标记。[73]中提出的标签平滑方法是将硬标签转换为软标签进行训练，使模型的鲁棒性更强。为了获得更好的软标签，Islam 等人引入了知识精馏的概念来设计标签细化网络

        最后 bag of freebies 是边界盒 (BBox) 回归的目标函数。传统的对象检测器通常使用均方误差 (MSE) 直接对 BBox 的中心点坐标和高度和宽度进行回归，{,w、h}或左上角点和右下角点。对于基于锚的方法，是估计相应的偏移量，例如和, 然而，直接估计 BBox 中每个点的坐标值是要将这些点作为自变量来处理，但实际上并没有考虑对象本身的完整性。为了更好地处理这一问题，一些研究人员最近提出了 IoU 损失 [90]，它考虑了预测的 BBox 区域和地面真实 BBox 区域的覆盖范围。IoU 损失计算过程将通过使用地面真相执行 IoU，触发 BBox 的四个坐标点的计算，然后将生成的结果连接到一个整个代码中。由于 IoU 是一种尺度不变表示，它可以解决传统方法计算{x、y、w、h} 的 l1 或 l2 损失时，损失会随着尺度的增加而增加的问题。最近，一些研究人员继续改善 IoU 的损失。例如，GIoU 损失 [65] 除了包括覆盖区域外，还包括物体的形状和方向。他们提出找到能够同时覆盖预测的 BBox 和地面真实 BBox 的最小面积的 BBox，并使用该 BBox 作为分母来代替 IoU 损失中最初使用的分母。对于 DIoU 损失 [99]，它另外考虑了物体中心的距离，而 CIoU 损失[99] 则同时考虑了重叠面积、中心点之间的距离和高宽比。CIoU 在 BBox 回归问题上可以获得更好的收敛速度和精度。

### 精读

#### BoF 方法一：数据增强

**（1）像素级调整**

**①光度失真：** brightness(亮度)、contrast(对比度)、hue(色度)、saturation(饱和度)、noise(噪声)

**②几何失真：** scaling(缩放尺寸)、cropping(裁剪)、flipping(翻转)、rotating(旋转)

**（2）模拟目标遮挡**

**①erase(擦除)、CutOut(剪切)：** 随机选择图像的矩形区域，并填充随机或互补的零值

**②hide-and-seek 和 grid mask：** 随机或均匀地选择图像中的多个矩形区域，并将它们替换为全零

**③将上述方式作用于特征图上：** DropOut、DropConnect、DropBlock

**（3）将多张图像组合在一起**

**①MixUp：** 使用两个图像以不同的系数比率相乘后叠加，利用叠加比率调整标签

**②CutMix：** 将裁剪的图像覆盖到其他图像的矩形区域，并根据混合区域大小调整标签

**（4）使用 style transfer GAN 进行数据扩充，有效减少 CNN 学习到的纹理偏差。**

#### BoF 方法二：解决数据集中语义分布偏差问题

**①两阶段对象检测器：** 使用硬反例挖掘或在线硬例挖掘来解决。不适用于单级目标检测。

**②单阶段目标检测器：** focal 损来处理各个类之间存在的数据不平衡问题。

#### BoF 方法三：边界框 (BBox) 回归的目标函数

**①IoU 损失：** 将预测 BBox 区域的区域和真实 BBox 区域考虑在内。由于 IoU 是尺度不变的表示，它可以解决传统方法在计算 {x, y, w, h} 的 l1 或 l2 损耗时，损耗会随着尺度的增大而增大的问题。

**②GIoU loss：** 除了覆盖区域外，还包括了物体的形状和方向。他们提出寻找能够同时覆盖预测 BBox 和地面真实 BBox 的最小面积 BBox，并以此 BBox 作为分母来代替 IoU 损失中原来使用的分母。

**③DIoU loss：** 它额外考虑了物体中心的距离。

**④CIoU loss ：** 同时考虑了重叠区域、中心点之间的距离和纵横比。对于 BBox 回归问题，CIoU 具有更好的收敛速度和精度。

### 2.3 Bag of specials

### 翻译

 对于那些只增加少量推理成本但又能显著提高目标检测精度的插件模块和后处理方法，我们称它们为 “bag of specials"。一般来说，这些插件模块是用于增强模型中的某些属性，如扩大接受域、引入注意机制或增强特征整合能力等，而后处理是筛选模型预测结果的一种方法。 

        可用于增强感受野的常见模块是 SPP[25]、ASPP[5]和 RFB[47]。SPP 模块起源于空间金字塔匹配 (SPM)[39]，SPMs 的原始方法是将特征映射分割成几个 d×d 相等的块，其中 d 可以是{1,2,3，…}，从而形成空间金字塔，然后提取 bag-of-word 特征。SPP 将 SPM 集成到 CNN 中，使用最大池化操作，而不是 bag-of-word 操作。由于 He 等人[25] 提出的 SPP 模块将输出一维特征向量，因此在全卷积网络 (FCN) 中应用是不可行的。因此，在 YOLOv3[63]的设计中，Redmon 和 Farhadi 将 SPP 模块改进为核大小为 k×k，其中 k={1,5,9,13}，步幅等于 1。在这种设计下，相对较大的最大池有效地增加了主干特征的接受域。 在添加改进版本的 SPP 模块后，YOLOv3-608 在 MS COCO 目标检测任务上将 AP50 升级了 2.7%，额外计算 0.5%。ASPP[5]模块与改进的 SPP 模块在操作上的差异主要是从原始的 k×k 核大小，步幅最大池化等于 1 到多个 3×3 核大小，扩张比等于 k，步幅等于 1。RFB 模块采用 k×k 核的多个扩张卷积，扩张比等于 k，步幅等于 1，以获得比 ASPP 更全面的空间覆盖。RFB[47]只需要花费 7% 的额外推理时间，就可以使 MS COCO 上的 SSD 的 AP50 增加 5.7%。

        目标检测中常用的注意模块主要分为通道式注意和点态注意，这两种注意模型的代表分别是 Squeeze-and-Excitation (SE)[29] 和空间注意模块 (SAM)[85]。虽然 SE 模块可以提高 ResNet50 的力量在 ImageNet 图像分类任务 1%top-1 精度的只增加 2% 计算，但在 GPU 通常将使推理时间增加约 10%，所以它更适合用于移动设备。但对于 SAM，它只需要额外支付 0.1% 的计算量，就可以将 ResNet50-SE 提高到 ImageNet 图像分类任务的 0.5% 的 top-1 精度。最重要的是，它根本不影响 GPU 上的推理速度。

        在特征集成方面，早期的实践是使用 skip connection[51] 或 hyper-column[22] 将低级物理特征与高级语义特征进行集成。随着 FPN 等多尺度预测方法越来越流行，人们提出了许多整合不同特征金字塔的轻量级模块。这类模块包括 SFAM[98]、ASFF[48] 和 BiFPN[77]。SFAM 的主要思想是利用 SE 模块在多尺度连接的特征图上执行信道级重加权。对于 ASFF，它使用 softmax 作为点级重新加权，然后添加不同尺度的特征图。在 BiFPN 中，提出了多输入加权残差连接来进行尺度水平重加权，然后添加不同尺度的特征图。

        在深度学习的研究中，一些人将重点放在寻找良好的激活函数上。一个好的激活函数可以使梯度更有效地传播，同时也不会造成太多的额外计算成本。2010 年，Nair 和 Hinton[56]提出 ReLU 来实质上解决传统的 tanh 和 s 型激活函数中经常遇到的梯度消失问题。随后，提出了 LReLU[54]、PReLU[24]、ReLU6[28]、尺度指数线性单位 (SELU)[35]、Swish[59]、hard-Swish[27]、Mish[55] 等，它们也被用于解决梯度消失问题。LReLU 和 PReLU 的主要目的是解决当输出小于零时，ReLU 的梯度为零的问题。对于 ReLU6 和 hard-swish，它们是专门为量化网络设计的。对于神经网络的自归一化，提出了 SELU 激活函数来满足该目标。需要注意的一点是，Swish 和 Mish 都是连续可区分的激活函数。 

        在基于深度学习的对象检测中常用的后处理方法是 NMS，它可以用于过滤那些预测同一对象不好的预测框，并且只保留响应率较高的候框。NMS 试图改进的方法与优化目标函数的方法是一致的。NMS 提出的原始方法不考虑上下文信息，因此 Girshick 等 [19] 在 R-CNN 中添加分类置信分数作为参考，根据置信分数的顺序，按照高到低的顺序进行 greedy NMS。对于 soft NMS[1]，它考虑了对象的遮挡在 greedy NMS 中可能导致置信度分数下降的问题。DIoU NMS[99]开发者的思维方式是在 soft NMS 的基础上，将中心点距离的信息添加到 BBox 的筛选过程中。值得一提的是，由于上述所有的后处理方法都没有直接涉及到所捕获的图像特征，因此在后续的无锚定方法的开发中，不再需要后处理。

### 精读

#### BoS 方法一：插件模块之增强感受野

**①改进的 SPP 模块**

![](https://img-blog.csdnimg.cn/576497d1e24447a8ae8be21da526e5be.png)

**②ASPP 模块**

![](https://img-blog.csdnimg.cn/8bd60831a6db40cb9143849e37b1124f.png)

**③RFB 模块**

![](https://img-blog.csdnimg.cn/c80fc02aeda244d39fa919d446c6e8a9.png)

#### BoS 方法二：插件模块之注意力机制

**①channel-wise 注意力：** 代表是 Squeeze-and-Excitation 挤压激励模块 (SE)。

![](https://img-blog.csdnimg.cn/5f01ce134acb4a57828228cdfecc80a5.png)

**②point-wise 注意力：** 代表是 Spatial Attention Module 空间注意模块 (SAM)。

![](https://img-blog.csdnimg.cn/f8e98059d54a47e4824c9c5db8861c5e.png)

#### BoS 方法三：插件模块之特征融合

**①SFAM：** 主要思想是利用 SE 模块在多尺度的拼接特征图上进行信道级重加权。

![](https://img-blog.csdnimg.cn/b6480eb81a4349b78462b593dc1da850.png)

**②ASFF：** 使用 softmax 对多尺度拼接特征图在点维度进行加权。

![](https://img-blog.csdnimg.cn/40c511b6dbd54e41b511b913660ace6f.png)

 **③BiFPN：** 提出了多输入加权剩余连接来执行按比例的水平重加权，然后添加不同比例的特征图。

![](https://img-blog.csdnimg.cn/9c3b6e13706d436080c87c925fdbf804.png)

#### BoS 方法四：激活函数

**①LReLU 和 PReLU：** 主要目的是解决输出小于 0 时 ReLU 的梯度为零的问题。

**②ReLU6 和 hard-Swish：** 专门为量化网络设计的。

**③SELU：** 针对神经网络的自归一化问题。

**④Swish 和 Mish：** 都是连续可微的激活函数。

#### BoS 方法五：后处理

**①NMS：** 目标检测中常用的后处理方法是 NMS, NMS 可以对预测较差的 bbox 进行过滤，只保留响应较高的候选 bbox。NMS 试图改进的方法与优化目标函数的方法是一致的。NMS 提出的原始方法没有考虑上下文信息，所以在 R-CNN 中加入了分类的置信分作为参考，按照置信分的顺序，从高到低依次进行贪心 NMS。

**②soft NMS：** 考虑了对象的遮挡可能导致带 IoU 分数的贪婪 NMS 的信心分数下降的问题。

**③DIoU NMS：** 在 soft NMS 的基础上，将中心点距离信息添加到 BBox 筛选过程中。值得一提的是，由于以上的后处理方法都没有直接引用捕获的图像特征，因此在后续的无锚方法开发中不再需要后处理。

**三、Methodology—方法**
--------------------

### 3.1 Selection of architecture—架构选择 

### 翻译 

  
        我们的目标是在输入网络分辨率、卷积层数、参数数（滤波器大小 2 * 滤波器 * 通道 / 组）和层输出数（滤波器）之间找到最优的平衡。例如，我们的大量研究表明，在 ILSVRC2012(ImageNet)数据集 [10] 上，CSPResNext50 比 CSPDarknet53 要好得多。然而，相反地，在检测 MS COCO 数据集 [46] 上的对象方面，CSPDarknet53 比 CSPResNext50 更好。

        下一个目标是选择额外的块来增加感受野，以及从不同检测器级别的参数聚合的最佳方法：例如 FPN、PAN、ASFF、BiFPN。 

        对于分类最优的参考模型对于探测器来说并不总是最优的。与分类器相比，该探测器需要以下条件：

*           更高的输入网络大小（分辨率）
*           用于检测多个小大小的物体更多的层
*          更高的接受域以覆盖增加的输入网络大小更多的参数
*          模型更大的能力来检测单一图像中多个不同大小的物体

        假设来说，我们可以假设应该选择一个具有更大的接受场大小（具有更多的卷积层 3×3）和更多的参数的模型作为主干。表 1 显示了 CSPResNeXt50、CSPDarknet53 和 efficientnetB3 的信息。CSPResNext50 只包含 16 个卷积层 3×3、一个 425×425 感受野和 20.6 M 参数，而 CSPDarknet53 包含 29 个卷积层 3×3、一个 725×725 感受野和 27.6 M 参数。这一理论证明，加上我们进行的大量实验，表明 CSPDarknet53 神经网络是两者作为探测器主干的最佳模型。 

        不同大小的感受野的影响总结如下：

*   到对象大小，允许查看整个对象到网络大小
*   允许查看对象周围的上下文
*   增加图像点和最终激活之间的连接数量 

        我们在 CSPDarknet53 上添加了 SPP 块，因为它显著地增加了接受域，分离出了最重要的上下文特征，并且几乎不会导致降低网络运行速度。我们使用 PANet 作为来自不同检测器级别的不同主干级别的参数聚合的方法，而不是在 YOLOv3 中使用的 FPN。

        最后，我们选择 CSPDarknet53 主干、SPP 附加模块、PANet 路径聚合颈和 YOLOv3（基于锚点）的头作为 YOLOv4 的体系结构。

        未来，我们计划显著扩展检测器的 f Bag of Freebies(BoF) 的内容，理论上可以解决一些问题，提高检测器的精度，并以实验方式依次检查每个特征的影响。

        我们不使用 Cross-GPU 批处理归一化 (CGBN 或 SyncBN) 或昂贵的专用设备。这允许任何人都可以在传统的图形处理器上再现我们最先进的结果，例如 GTX 1080Ti 或 RTX 2080Ti。   
 

### 精读

#### 架构选择目标

**目标一：在输入网络分辨率、卷积层数、参数数 (filter size2×filters × channel / groups) 和层输出数 (filters) 之间找到最优平衡。**

检测器需要满足以下条件：

**①更高的输入网络大小 (分辨率)：** 用于检测多个小型对象

**②更多的层：** 一个更高的接受域，以覆盖增加的输入网络的大小

**③更多的参数：** 模型有更强大的能力，以检测单个图像中的多个不同大小的对象。

**目标二：选择额外的块来增加感受野**

不同大小的感受野的影响总结如下：

**①对象大小：** 允许查看整个对象

**②网络大小：** 允许查看对象周围的上下文

**③超过网络大小：** 增加图像点和最终激活之间的连接数

**目标三：选择不同的主干层对不同的检测器层 (如 FPN、PAN、ASFF、BiFPN) 进行参数聚合的最佳方法。**

#### YOLOv4 架构

**（1）CSPDarknet53 主干（backbone）：** 作者实验对比了 CSPResNext50、CSPDarknet53 和 EfficientNet-B3。从理论与实验角度表明：CSPDarkNet53 更适合作为检测模型的 Backbone。（还是自家的网络结构好用）

![](https://img-blog.csdnimg.cn/315ea8719c9c4309b67e42c6a0822bbf.png)

> CSP 介绍：
> 
> CSP 是可以增强 CNN 学习能力的新型 backbone，论文发表 2019 年 11 月份
> 
> **主要技巧：**CSPNet 将底层的特征映射分为两部分，一部分经过密集块和过渡层，另一部分与传输的特征映射结合到下一阶段。

**（2）SPP 附加模块增加感受野：** 在 CSPDarknet53 上添加了 SPP 块，SPP 来源于何恺明大佬的 SPP Net 因为它显著增加了接受域，分离出了最重要的上下文特性，并且几乎不会降低网络运行速度。

**（3）PANet 路径聚合（neck）：** PANet 主要是特征融合的改进，使用 PANet 作为不同检测层的不同主干层的参数聚合方法。而不是 YOLOv3 中使用的 FPN。

**（4）基于锚的 YOLOv3 头部（head）：** 因为是 anchor-base 方法，因此分类、回归分支没有改变。

**总结：** YOLOv4 模型 = CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head

### 3.2 Selection of BoF and BoS—BoF 和 BoS 的选择

### 翻译

为了改进目标检测训练，CNN 通常使用以下：

*   激活：ReLU, leaky-ReLU, parametric-ReLU,ReLU6, SELU, Swish, or Mish
*   边界盒回归损失：MSE，IoU、GIoU、CIoU、DIoU
*   数据增强：CutOut, MixUp, CutMix
*   正则化方法：DropOut, DropPath，Spatial DropOut [79], or DropBlock
*   规范化的网络激活（通过均值和方差）：批标准化 (BN)[32]，Cross-GPU Batch Normalization(CGBN 或 SyncBN)[93]，Filter Response Normalization(FRN)[70]，或交叉迭代批标准化 (CBN)[89]
*   Skip-connections：Residual connections，加权 Residual connections、多输入加权 Residual connections 或 Cross stage partial 连接 (CSP) 

        对于训练激活函数，由于 PReLU 和 SELU 更难训练，而且 ReLU6 是专门为量化网络设计的，因此我们将上述激活函数从候选列表中删除。在需求化方法上，发表 DropBlock 的人将其方法与其他方法进行了详细的比较，其正则化方法获得了很大的成功。因此，我们毫不犹豫地选择了 DropBlock 作为我们的正则化方法。至于归一化方法的选择，由于我们关注于只使用一个 GPU 的训练策略，因此不考虑 syncBN。   
 

### 精读

为了提高目标检测训练，CNN 通常使用以上提到的方法（具体在[【YOLO 系列】YOLOv4 论文超详细解读 2（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129248856 "【YOLO系列】YOLOv4论文超详细解读2（网络详解）")里详细讲解）

**（1）激活函数：** 由于 PReLU 和 SELU 更难训练，我们选择专门为量化网络设计的 ReLU6

**（2）正则化：** 我们选择 DropBlock

**（3）归一化：** 由于是单 GPU，所以没有考虑 syncBN

### 3.3 Additional improvements—进一步改进

### 翻译

为了使设计的探测器更适合训练单 GPU 上，我们做了额外的设计和改进如下：

 我们引入了一种新的数据增强 Mosic，和自我对抗训练（SAT）  
我们选择最优超参数而应用遗传算法  
我们修改一些现有方法使设计适合有效的训练和检测，modifified SAM，modifified PAN，和交叉小批归一化 (CmBN)   
        Mosaic 代表了一种新的数据增强方法，它混合了 4 个训练图像。因此，混合了 4 种不同的上下文，而 CutMix 只混合了 2 个输入图像。这允许检测其正常上下文之外的对象。此外，批归一化计算每一层上 4 个不同图像的激活统计信息。这大大减少了对大型小批量处理的需求

        自对抗训练 (SAT) 也代表了一种新的数据增强技术，可以在 2 个向前向后的阶段运行。在第一阶段，神经网络改变了原始图像，而不是网络的权值。通过这种方式，神经网络对自己进行敌对性攻击，改变原始图像，以制造出图像上没有想要的目标的欺骗。在第二阶段，神经网络被训练以正常的方式检测修改后的图像上的对象。   
 

### 精读

#### （1）新的数据增强 Mosic 和自我对抗训练（SAT）

**①Mosaic：** Mosaic 代表了一种新的数据增强方法，它混合了 4 幅训练图像。基于现有数据极大的丰富了样本的多样性，极大程度降低了模型对于多样性学习的难度。

**②自对抗训练（SAT）：**

*   在第一阶段，神经网络改变原始图像而不是网络权值。通过这种方式，神经网络对自己执行一种对抗性攻击，改变原始图像，以制造图像上没有期望对象的假象。
*   在第二阶段，神经网络以正常的方式对这个修改后的图像进行检测。

#### （2）应用遗传算法选择最优超参数

#### （3）修改现有的方法，使设计适合于有效的训练和检测

**①修改的 SAM：** 将 SAM 从空间上的注意修改为点态注意

![](https://img-blog.csdnimg.cn/d76725327f3540b2becd085723b44a60.png)

**②修改 PAN：** 将 PAN 的快捷连接替换为 shortcut 连接

![](https://img-blog.csdnimg.cn/22ee40acc9384f8382762b026673f5b3.png)

**③交叉小批量标准化 (CmBN)：** CmBN 表示 CBN 修改后的版本，如图所示，只在单个批内的小批之间收集统计信息。

![](https://img-blog.csdnimg.cn/d133bdbd7bcb4abc94891de303612741.png)

### 3.4 YOLOv4

### 翻译

 在本节中，我们将详细说明 YOLOv4 的细节。

YOLOv4 consists of :  
• Backbone: CSPDarknet53 [81]  
• Neck: SPP [25], PAN [ 49 ]  
• Head: YOLOv3 [63]  
![](https://img-blog.csdnimg.cn/be21fe3ae67841b3843ae7c899520e8f.png)

### 精读

#### YOLOv4 包括

*   **主干 (backbone)：** CSPDarknet53
*   **颈部 (neck)：** SPP ， PAN
*   **头 (head)：** YOLOv3

#### YOLO v4 使用

*   **Bag of Freebies 外在引入技巧：** CutMix 和 Mosaic 数据增强，DropBlock 正则化，类标签平滑
*   **Bag of Specials 网络改进技巧：** Mish 激活、跨级部分连接 (CSP)、多输入加权剩余连接 (MiWRC)
*   **Bag of Freebies 外在检测器引入技巧：** CIoU loss, CmBN, DropBlock 正则化，Mosaic 数据增强，自对抗训练，消除网格敏感性，为一个真值使用多个锚，余弦退火调度，最优超参数，随机训练形状
*   **Bag of Specials 检测器网络改进技巧：** Mish 激活、SPP-block、SAM-block、PAN 路径聚合块、DIoU-NMS

**四、Experiments—实验**
--------------------

### 4.1 Experimental setup—实验设置

### 翻译

 在 ImageNet 图像分类实验中，默认的超参数如下：训练步骤为 8000000；批大小和小批量大小分别为 128 和 32；采用多项式衰减学习率调度策略，初始学习率为 0.1；预热步骤为 1000；动量衰减和权重衰减分别设置为 0.9 和 0.005。我们所有的 BoS 实验都使用与默认设置相同的超参数，在 BoF 实验中，我们额外添加了 50% 的训练步骤。在 BoF 实验中，我们验证了 MixUp、CutMix、Mosaic、模糊数据增强和标签平滑正则化方法。在 BoS 实验中，我们比较了 LReLU、Swish 和 Mish 激活功能的影响。所有实验均采用 1080 Ti 或 2080TiGPU 进行训练。 

         在 MS COCO 目标检测实验中，默认的超参数如下：训练步长为 500,500；采用步长衰减学习率调度策略，初始学习率为 0.01，在 400000 步和 450000 步时分别乘以 0.1 倍；动量和权重衰减分别设置为 0.9 和 0.0005。所有架构都使用一个 GPU 来执行批处理大小为 64 的多规模训练，而小批处理大小为 8 或 4，这取决于架构和 GPU 内存限制。除在超参数搜索实验中使用遗传算法外，所有其他实验均使用默认设置。遗传算法使用 YOLOv3-SPP 对 GIoU 损失进行训练，并搜索 300 个时元的最小值 5k 集。我们采用搜索学习率 0.00261，动量 0.949，IoU 阈值分配地面真值 0.213，遗传算法实验采用损失归一化器 0.07。我们验证了大量的 BoF，包括网格灵敏度消除、Mosaic 数据增强、IoU 阈值、遗传算法、类标签平滑、交叉小批归一化、自对抗训练、余弦退火调度器、动态小批大小、dropblock、优化锚点、不同类型的 IoU 损失。我们还在各种 BoS 上进行了实验，包括 Mish、SPP、SAM、RFB、BiFPN 和高斯 YOLO[8]。对于所有的实验，我们只使用一个 GPU 来进行训练，因此不使用优化多个 GPU 的像 syncBN 这样的技术。

### 精读

（1）在 ImageNet 图像分类实验中，默认超参数为：

*    **训练步骤：** 8,000,000
*    **批大小和小批大小分别：** 128 和 32
*    **初始学习率：** 0.1
*    **warm-up 步长：** 1000
*    **动量衰减：** 0.9
*    **权重衰减：** 0.005

（2）在 MS COCO 对象检测实验中，默认的超参数为：

*    **训练步骤：** 500500
*    **初始学习率：** 0.01
*    **warm-up 步长：** 在 400,000 步和 450,000 步分别乘以因子 0.1
*    **动量衰减：** 0.9
*    **权重衰减：** 0.0005
*    **GPU 数量：** 1 个
*    **批处理大小：** 64

### 4.2 Influence of different features on Classifier training—不同特征对分类器训练的影响

### 翻译

        首先，我们研究了不同特征对分类器训练的影响；具体来说，类标签平滑的影响，不同数据增强技术的影响，bilateral blurring, MixUp, CutMix and Mosaic，如 Fugure7 所示，以及不同激活的影响，如 Leaky-relu（默认）、Swish 和 Mish。 

        在我们的实验中，如表 2 所示，通过引入：CutMix 和 Mosaic 数据增强、类标签平滑和 Mish 激活等特征，提高了分类器的精度。因此，我们用于分类器训练的 BoF backbone(Bag of Freebies) 包括以下内容：CutMix 和 Mosaic 数据增强和类标签平滑。此外，我们使用 Mish 激活作为补充。 

### 精读

研究了不同特征对分类器训练的影响：类标签平滑的影响，不同数据增强技术的影响，不同的激活的影响。

 ![](https://img-blog.csdnimg.cn/dbd77eaa81094b02b758aeef7f451f14.png)

 图 7：各种数据增强方法

![](https://img-blog.csdnimg.cn/84257f1d80684f13a2536bdf58cffa44.png)

表 2：Bof 和 Mish 对 Cspresnext - 50 Clas - Sifier 准确率的影响

![](https://img-blog.csdnimg.cn/41be7b33b64a43c3abf37a9198a91d85.png)

表 3：Bof 和 Mish 对 Cspdarknet - 53 Classi - Fier 精度的影响

#### 结论

（1）通过引入特征如：CutMix 和 Mosaic 数据增强、类标签平滑、Mish 激活等，可以提高分类器的准确率。

（2）CutMix 和 Mosaic 数据增强和类标签平滑可用于分类器训练的 BoF backbone，此外，还可以使用 Mish 激活作为补充选项。

### **4.3 Influence of different features on Detector training—不同特征对检测器训练的影响**

### 翻译

 进一步研究了不同的 Bag-of Freebies(BoF-detector) 对探测器训练精度的影响，如表 4 所示。通过研究在不影响 FPS 的情况下提高检测器精度的不同特征，我们显著地扩展了 BoF 列表： 

        S：消除网格灵敏度的公式 其中 cx 和 cy 总是整数，在 YOLOv3 中用于计算对象坐标，因此，对于接近 cx 或 cx+1 值的 bx 值，需要极高的 tx 绝对值。我们通过将 s 型矩阵乘以一个超过 1.0 的因子来解决这个问题，从而消除了对象无法检测到的网格的影响。

*           M：Mosaic data - 在训练期间使用 4 张图像的马赛克，而不是单一的图像 
*           IT：IoU 阈值 - 使用多个锚作为单一地面真实 IoU(truth, anchor) >IoU 阈值
*           GA：Genetic algorithms - 在前 10% 的时间段内使用遗传算法选择最优超参数
*           LS: 类标签平滑 - 使用类标签平滑的 s 型符号激活 
*           CBN：CmBN - 使用交叉小批标准化来收集整个批内的统计信息，而不是在单个小批内收集统计数据
*           CA: 余弦退火调度器 - 改变正弦波训练过程中的学习速率
*           DM：动态小批量大小 - 在小分辨率训练中，通过使用随机训练形状自动增加小批量大小
*           OA：优化的锚 - 使用优化的锚与 512x512 网络分辨率进行训练
*           GIoU，CIoU，DIoU，MSE - 使用不同的损失算法进行边界框回归 

        进一步研究了不同的 Bag-of-Specials (bos - 检测器) 对检测器训练精度的影响，包括 PAN、RFB、SAM、高斯 YOLO(G) 和 ASFF，如表 5 所示。在我们的实验中，检测器在使用 SPP、PAN 和 SAM 时性能最好。

### 精读

进一步的研究关注不同 Bag-of-Freebies 免费包 (BoF-detector) 对检测器训练精度的影响，通过研究在不影响 FPS（帧率：每秒传输的帧数）的情况下提高检测器精度的不同特征，我们显著扩展了 BoF 列表：

![](https://img-blog.csdnimg.cn/478c708905f14c568c032e6307e9f0f4.png)

表 4：Bag-of-Freebies 的消融研究。(CSPResNeXt50 - PANet - SPP , 512 × 512)。 粗体黑色表示有效

![](https://img-blog.csdnimg.cn/a99079cd475745988d8ff33276e28d49.png)

表 5：Bag-of-Specials 的消融研究。( 512 × 512 ）

#### 结论

当使用 SPP、PAN 和 SAM 时，检测器的性能最佳。

### 4.4 Influence of different backbones and pre- trained weightings on Detector training—不同的 backbone 和预先训练权重对检测器训练的影响

### 翻译

进一步研究了不同主干模型对检测器精度的影响，如表 6 所示。我们注意到，具有最佳分类精度特征的模型在检测器精度方面并不总是最好的。

         首先，虽然使用不同特征训练的 CSPResNeXt-50 模型的分类精度高于 CSPDarknet53 模型，但 CSPDarknet53 模型在目标检测方面具有更高的精度。

        其次，使用 CSPResF 和 Mish 进行 50 分类器训练可以提高分类精度，但进一步应用这些预先训练的权重用于检测器训练会降低检测器的精度。然而，在 CSPDarknet53 分类器训练中使用 BoF 和 Mish 可以提高分类器和使用该分类器预训练的加权的检测器的准确性。最终的结果是，主干 CSPDarknet53 比 CSPResNeXt50 更适合用于检测器。

        我们观察到，CSPDarknet53 模型由于各种改进，显示出更大的能力来提高探测器的精度。

### 精读

![](https://img-blog.csdnimg.cn/1308751ebf7f46c5839045366f9ebf79.png)

表 6：使用不同的分类器预训练权重进行检测器训练 (所有其他训练参数在所有模型中都是相似的)。

#### 结论

*   具有最佳分类精度的模型在检测器精度方面并不总是最佳的。

*   骨干 CSPDarknet53 比 CSPResNeXt50 更适合于检测器。
*   由于各种改进，CSPDarknet53 模型展示了更大的能力来提高检测器的精度。

### 4.5 Influence of different mini-batch size on Detec- tor training—不同的小批尺寸对检测器培训的影响

### 翻译

最后，我们分析了用不同的小批量训练的模型得到的结果，结果如表 7 所示。从表 7 所示的结果中，我们发现在添加 BoF 和 BoS 训练策略后，小批量大小对检测器的性能几乎没有影响。这一结果表明，在引入 BoF 和 BoS 后，不再需要使用昂贵的 gpu 进行训练。换句话说，任何人都只能使用一个普通的 GPU 来训练一个优秀的探测器。

### 精读

![](https://img-blog.csdnimg.cn/02a0eed7169844fcbff42aeb875b1f9e.png)

表 7：使用不同的 mini-batch size 进行检测器训练。

#### 结论

*   加入 BoF 和 BoS 训练策略后，小批量大小对检测器的性能几乎没有影响。
*   minibatch 越大越好，CSPDarknet53 对 minibatch 不敏感，利于单卡训练。
*   在引入 BoF 和 BoS 之后，不再需要使用昂贵的 GPU 进行训练。

 五、Results—结果
-------------

### 翻译

与其他最先进的对象检测器所获得的结果的比较如图 8 所示。我们的 YOLOv4 位于 Pareto 最优性曲线上，在速度和精度方面都优于最快和最精确的探测器。 

        由于不同的方法使用不同架构的 gpu 进行推理时间验证，我们在通常采用的 Maxwell、Pascal 和 Volta 架构的 gpu 上操作 YOLOv4，并将它们与其他最先进的方法进行比较。表 8 列出了使用 MaxwellGPU 的帧率比较结果，它可以是 GTX TitanX（Maxwell）或 TeslaM40GPU。表 9 列出了使用 PascalGPU 的帧率比较结果，可以是 TitanX(Pascal)、TitanXp、GTX 1080Ti 或特斯拉 P100GPU。如表 10 所述，它列出了使用 VoltaGPU 的帧率比较结果，可以是 Titan Volta 或 Tesla V100GPU。 

### 精读

![](https://img-blog.csdnimg.cn/a91039d060054f99a1f28fa728168d49.png)

图 8 不同物体探测器的速度和精度比较。(一些文章只针对其中一个 GPU : Maxwell / Pascal / Volta ，阐述了它们探测器的 FPS)

#### 结论

*   得到的结果与其他最先进的物体探测器的比较如图 8 所示。我们的 YOLOv4 位于 Pareto 最优曲线上，无论是速度还是精度都优于最快最准确的检测器。
*   由于不同的方法使用不同架构的 gpu 进行推理时间验证，我们在 Maxwell 架构、Pascal 架构和 Volta 架构常用的 gpu 上运行 YOLOv4，并与其他最先进的方法进行比较。

这篇就是论文的解读，因为涉及到太多 tricks 我目前理解的也不够深，以后再慢慢补充吧~