> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129248856?spm=1001.2014.3001.5501)

![](https://img-blog.csdnimg.cn/d84d43d9185d4cc2986d4a74a38c028f.gif)

 上一篇我们一起读了 [YOLOv4](https://so.csdn.net/so/search?q=YOLOv4&spm=1001.2101.3001.7020) 的论文《YOLOv4：Optimal Speed and Accuracy of Object Detection》（直通车→[【YOLO 系列】YOLOv4 论文超详细解读 1（翻译 ＋学习笔记）](https://blog.csdn.net/weixin_43334693/article/details/129232468?spm=1001.2014.3001.5501 "【YOLO系列】YOLOv4论文超详细解读1（翻译 ＋学习笔记）")），有了初步的印象，论文里面涉及到很多 tricks，上一篇介绍的比较简略，我们这篇来详细介绍一下。

**目录**
------

[一、YOLOv4 的简介](#%E4%B8%80%E3%80%81YOLOv4%E7%9A%84%E7%AE%80%E4%BB%8B)

 [二、YOLOv4 的网络结构](#t0)

[三、输入端](#t1)

[数据增强①CutMix](#t2)

[数据增强②Mosaic](#t3)

[SAT 自对抗训练](#t4)

[cmBN](#t5)

[Label Smoothing 类标签平滑](#t6)

[四、主干网络 BackBone](#t7)

[CSPDarknet53](#t8)

[Mish 激活函数](#t9)

[Dropblock 正则化](#t10)

[五、Neck](#t11)

[SPP](#t12)

[PAN](#t13)

[SAM](#t14)

[六、Head](#t15)

[Loss](#t16)

[NMS](#t17)

一、YOLOv4 的简介
------------

**YOLOv4 一共有如下三点贡献：**

（1）开发了一个高效、强大的目标检测模型。它使每个人都可以使用 1080ti 或 2080ti GPU 来训练一个非常快速和准确的目标检测器。

（2）验证了最先进的 Bag-of-Freebies 和 Bag-of-Specials 对象检测在检测器训练时的影响。

（3）对现有的方法进行了改进，使其更加高效，更适合于单个 GPU 的训练，包括 CBN，PAN，SAM 等。

![](https://img-blog.csdnimg.cn/8899d178c3a344a2a271272dd5d6e707.png)

 二、YOLOv4 的网络结构
---------------

YOLOv4 的整体原理图如下：和 v3 还是比较接近的

![](https://img-blog.csdnimg.cn/1273a31b84514074ac812b3d65bc6534.png)

可以看到由以下四个部分组成：

**输入端：** 训练时对输入端的改进，主要包括 Mosaic 数据增强、cmBN、SAT 自对抗训练

**BackBone 主干网络：** 各种方法技巧结合起来，包括：CSPDarknet53、Mish 激活函数、Dropblock

**Neck：** 目标检测网络在 BackBone 和最后的输出层之间往往会插入一些层，比如 YOLOv4 中的 SPP 模块、FPN+PAN、SAM 结构

**Head：** 输出层的锚框机制和 YOLOv3 相同，主要改进的是训练时的回归框位置损失函数 CIOU Loss，以及预测框筛选的 nms 变为 DIOU nms

YOLOv4 的五个基本组件：

1.  **CBM：**Yolov4 网络结构中的最小组件，由 Conv+Bn+Mish 激活函数三者组成。
2.  **CBL：**由 Conv+Bn+Leaky_relu 激活函数三者组成。
3.  **Res unit：**借鉴 Resnet 网络中的残差结构，让网络可以构建的更深。
4.  **CSPX：**借鉴 CSPNet 网络结构，由三个卷积层和 X 个 Res unint 模块 Concate 组成。
5.  **SPP：**采用 1×1，5×5，9×9，13×13 的最大池化的方式，进行多尺度融合。

三、输入端
-----

Yolov4 对训练时的输入端进行改进，使得训练时在单张 GPU 上跑的结果也蛮好的。比如数据增强 Mosaic、cmBN、SAT 自对抗训练。

### 数据增强①CutMix

**数据增强的原因：**在平时项目训练时，小目标的 AP 一般比中目标和大目标低很多。而 Coco 数据集中也包含大量的小目标，但比较麻烦的是小目标的分布并不均匀。Coco 数据集中小目标占比达到 41.4%，数量比中目标和大目标都要多。但在所有的训练集图片中，只有 52.3% 的图片有小目标，而中目标和大目标的分布相对来说更加均匀一些。

**核心思想：**将一部分区域 cut 掉但不填充 0 像素，而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配。

**处理方式：**对一对图片做操作，随机生成一个裁剪框 Box，裁剪掉 A 图的相应位置，然后用 B 图片相应位置的 ROI 放到 A 图中被裁剪的区域形成新的样本，ground truth 标签会根据 patch 的面积按比例进行调整。

![](https://img-blog.csdnimg.cn/9b2449624d5a4e30b46e42350b5b0a15.png)

另外两种数据增强的方式：

**（1）Mixup:** 将随机的两张样本按比例混合，分类的结果按比例分配

**（2）Cutout:** 随机的将样本中的部分区域 Cut 掉，并且填充 0 像素值，分类的结果不变

### 数据增强②Mosaic

Yolov4 中使用的 Mosaic 是参考 2019 年底提出的 CutMix 数据增强的方式，但 CutMix 只使用了两张图片进行拼接，而 Mosaic 数据增强则采用了 4 张图片，随机缩放、随机裁剪、随机排布的方式进行拼接。

![](https://img-blog.csdnimg.cn/aa46625d25404dab91c58a152345f594.png)

**优点：**

**（1）丰富数据集：** 随机使用 4 张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好。

**（2）batch 不需要很大：** Mosaic 增强训练时，可以直接计算 4 张图片的数据，使得 Mini-batch 大小并不需要很大，一个 GPU 就可以达到比较好的效果。

### SAT 自对抗训练

自对抗训练 (SAT) 也代表了一种新的数据增加技术，在两个前后阶段操作。

**（1）在第一阶段：** 神经网络改变原始图像而不是网络权值。通过这种方式，神经网络对自己执行一种对抗性攻击，改变原始图像，以制造图像上没有期望对象的假象。

**（2）在第二阶段：** 神经网络以正常的方式对这个修改后的图像进行检测。

通过引入噪音点进行数据增强

![](https://img-blog.csdnimg.cn/ca557aa314c348a190ae9239694cbb13.png)

### cmBN

**BN：** 无论每个 batch 被分割为多少个 mini batch，其算法就是在每个 mini batch 前向传播后统计当前的 BN 数据（即每个神经元的期望和方差）并进行 Nomalization，BN 数据与其他 mini batch 的数据无关。

**CBN：** 每次 iteration 中的 BN 数据是其之前 n 次数据和当前数据的和（对非当前 batch 统计的数据进行了补偿再参与计算），用该累加值对当前的 batch 进行 Nomalization。好处在于每个 batch 可以设置较小的 size。

**CmBN：** 只在每个 Batch 内部使用 CBN 的方法，若每个 Batch 被分割为一个 mini batch，则其效果与 BN 一致；若分割为多个 mini batch，则与 CBN 类似，只是把 mini batch 当作 batch 进行计算，其区别在于权重更新时间点不同，同一个 batch 内权重参数一样，因此计算不需要进行补偿。

![](https://img-blog.csdnimg.cn/35e6d2ab061344969de54b5e942e6401.png)

### Label Smoothing 类标签平滑

**原因：**对预测有 100% 的信心可能表明模型是在记忆数据，而不是在学习。如果训练样本中会出现少量的错误样本，而模型过于相信训练样本，在训练过程中调整参数极力去逼近样本，这就导致了这些错误样本的负面影响变大。

**具体做法：**标签平滑调整预测的目标上限为一个较低的值，比如 0.9。它将使用这个值而不是 1.0 来计算损失。这样就缓解了过度拟合。说白了，这个平滑就是一定程度缩小 label 中 min 和 max 的差距，label 平滑可以减小过拟合。所以，适当调整 label，让两端的极值往中间凑凑，可以增加泛化性能。

![](https://img-blog.csdnimg.cn/b49987db314f48d0bb063b10cb9b9e6e.png)

 ![](https://img-blog.csdnimg.cn/93fe83994e6542129d7adfe36fd23a3c.png)

四、主干网络 BackBone
---------------

### CSPDarknet53

**简介：**CSPNet（Cross Stage Partial Networks），也就是跨阶段局部网络。CSPNet 解决了其他大型卷积神经网络框架 Backbone 中网络优化的梯度信息重复问题，CSPNet 的主要目的是使网络架构能够实现获取更丰富的梯度融合信息并降低计算量。

**具体做法：**CSPNet 实际上是基于 Densnet 的思想，即首先将数据划分成 Part 1 和 Part 2 两部分，Part 2 通过 dense block 发送副本到下一个阶段，接着将两个分支的信息在通道方向进行 Concat 拼接，最后再通过 Transition 层进一步融合。CSPNet 思想可以和 ResNet、ResNeXt 和 DenseNet 结合，目前主流的有 CSPResNext50 和 CSPDarknet53 两种改造 Backbone 网络。

![](https://img-blog.csdnimg.cn/0ed12e957eb14f45945cfc9f6d33a415.png)

**具体改进点：**

①用 Concat 代替 Add，提取更丰富的特征。

②引入 transition layer （1 * 1conv + 2 * 2pooling），提取特征，降低计算量，提升速度。

③将 Base layer 分为两部分进行融合，提取更丰富的特征。

> Q：为什么要采用 CSP 模块呢？
> 
> **CSPNet** 全称是 Cross Stage Paritial Network，主要从网络结构设计的角度解决推理中计算量很大的问题。
> 
> CSPNet 的作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的。
> 
> 因此采用 CSP 模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时，可以保证准确率。
> 
> 因此 YOLOv4 在主干网络 Backbone 采用 CSPDarknet53 网络结构，主要有三个方面的有点：
> 
> *   优点一：增强 CNN 的学习能力，使得在轻量化的同时保持准确性。
> *   优点二：降低计算瓶颈
> *   优点三：降低内存成本

### Mish 激活函数

**简介：**Mish 是一个平滑的曲线，平滑的激活函数允许更好的信息深入神经网络，从而得到更好的准确性和泛化；在负值的时候并不是完全截断，允许比较小的负梯度流入。Mish 是一个与 ReLU 和 Swish 非常相似的激活函数，但是 Relu 在小于 0 时完全杀死了梯度，不太符合实际情况，所以可以在不同数据集的许多深度网络中胜过它们。

**公式：**y=x∗tanh(ln(1+ex))

**Mish 图像：**

![](https://img-blog.csdnimg.cn/198bbe71b74e489193e24703d3c03aa7.png)

Mish 和 Leaky_relu 激活函数的图形对比如下：

![](https://img-blog.csdnimg.cn/9c216cc1c62242ceaa2ce0cb1049b54f.png)

**优点：**

（1）从图中可以看出该激活函数，在负值时并不是完全截断，而允许比较小的负梯度流入从而保证了信息的流动

（2）Mish 激活函数无边界，这让他避免了饱和（有下界，无上界）且每一点连续平滑且非单调性，从而使得梯度下降更好。

### Dropblock 正则化

**传统的 Dropout：**随机删除减少神经元的数量，使网络变得更简单。

**Dropblock：**DropBlock 技术在称为块的相邻相关区域中丢弃特征。Dropblock 方法的引入是为了克服 Dropout 随机丢弃特征的主要缺点，Dropout 主要作用在全连接层，而 Dropblock 可以作用在任何卷积层之上。这样既可以实现生成更简单模型的目的，又可以在每次训练迭代中引入学习部分网络权值的概念，对权值矩阵进行补偿，从而减少过拟合。

之前的 Dropout 是随机选择点 (b)，现在随机选择一个区域：

![](https://img-blog.csdnimg.cn/d4322a1b7f49443a9934a243f3b1ebf6.png)

> Q：全连接层上效果很好的 Dropout 在卷积层上效果并不好？
> 
>         中间 Dropout 的方式会随机的删减丢弃一些信息，但 Dropblock 的研究者认为，卷积层对于这种随机丢弃并不敏感，因为卷积层通常是三层连用：卷积 + 激活 + 池化层，池化层本身就是对相邻单元起作用。
> 
>         而且即使随机丢弃，卷积层仍然可以从相邻的激活单元学习到相同的信息。因此，在全连接层上效果很好的 Dropout 在卷积层上效果并不好。所以右图 Dropblock 的研究者则干脆整个局部区域进行删减丢弃。

五、Neck
------

### SPP

**简介：**SPP-Net 全称 Spatial Pyramid Pooling Networks，是何恺明大佬提出的，主要是用来解决不同尺寸的特征图如何进入全连接层的，在网络的最后一层 concat 所有特征图，后面能够继续接 CNN 模块。

如下图所示，下图中对任意尺寸的特征图直接进行固定尺寸的池化，来得到固定数量的特征。

![](https://img-blog.csdnimg.cn/930049684f2b4531852987dd2cc58a1c.png)

**具体结构如下：**

![](https://img-blog.csdnimg.cn/a786cbbbe5d7452681c7576a6b098c08.png)

### PAN

YOLOv3 中的 neck 只有自顶向下的 FPN，对特征图进行特征融合，而 YOLOv4 中则是 **FPN+PAN** 的方式对特征进一步的融合。引入了自底向上的路径，使得底层信息更容易传到顶部

下面是 YOLOv3 的 neck 中的 FPN，如图所示：

FPN 是自顶向下的，将高层的特征信息通过上采样的方式进行传递融合，得到进行预测的特征图。

![](https://img-blog.csdnimg.cn/49ec2b41a6094cccb7ddbc54461cdb62.png)

YOLOv4 中的 neck 如下：

![](https://img-blog.csdnimg.cn/9892b3a5389b4d0b9581e4bda8e4149c.png)

YOLOv4 在原始 PAN 结构上进行了一点改进，原本的 PANet 网络的 PAN 结构中，特征层之间融合时是直接通过 addition 的方式进行融合的，而 Yolov4 中则采用在通道方向 concat 拼接操作融合的，如下图所示。

![](https://img-blog.csdnimg.cn/81d999148b6b4356b8678b22caa7a8f7.png)

> Q：为什么要把 add 改为 concat？
> 
> **add：** 将两个特征图直接相加，是 resnet 中的融合方法，基于这种残差堆叠相加，可以有效地减小因为网络层数加深而导致的 cnn 网络退化问题。add 改变特征图像素值，并没有完全保留原本特征图信息，更多的可以看作对原特征图信息的一种补充，深层特征图在卷积过程中丢失了许多细节信息，通过 add 的方式得以补全，是在二维的平面上对特征图的增强。因此 add 在进行图像特征增强时使用最佳。
> 
> **concat：** 将两个特征图在通道数方向叠加在一起，原特征图信息完全保留下来，再对原特征图增加一些我们认为是较好的特征图，丰富了特征图的多样性，是在空间上对原特征图的增强，这样在下一次卷积的过程中我们能得到更好的特征图。

### SAM

SAM 源自于论文 CBAM(Convolutional Block Attention Module) 的论文，提出了两种注意力机制的技巧。

先来介绍一下 **CBAM**

如下图所示，输入一个特征 F，先进行 Channel attention module 后得到权重系数和原来的特征 F 相乘，然后在进行 Spatial attention module 后得到权重系数和原来的特征 F 相乘，最后就可以得到缩放后的新特征。不仅每个通道有注意力，而且特征图每个位置有注意力。

![](https://img-blog.csdnimg.cn/995c9147abc74e6fa1976687f2f5afb5.png)

接着我们来介绍 **Channel attention module(通道注意力模块)**

该模块就是将输入的特征 F 分别进行全局的 Maxpooling 与 Averagepooling，接着将这两个输入到一个权重共享的 MLP，再将这两个进行 element-wise summation 操作后经过 Sigmoid 函数会得到权重系数 Mc，再将这个权重系数与原来的特征 F 相乘, 就可以得到缩放后的新特征。

![](https://img-blog.csdnimg.cn/2f794329a11b4575962594cd454105e6.png)我们再看看 **Spatial attention module(空间注意力模块)**

首先对不同的 feature map 上相同位置的像素值进行全局的 Maxpooling 与 Average pooling，接着将这两个 spatial attention map 进行 concat，再利用一个 7X7 的卷积后经过 Sigmoid 函数会得到权重系数 Ms，在将这个权重系数与原来的特征 F 相乘，就可以得到缩放后的新特征，如下所示：

![](https://img-blog.csdnimg.cn/1fc4cb088bcc442d86654a4786dc1231.png)

YOLOv4 将 SAM 从空间注意修改为点注意，不应用最大值池化和平均池化，而是直接接一个 7X7 的卷积层，这样使速度相对快一些。

![](https://img-blog.csdnimg.cn/c86f3ed5780e44b5ab7c1c6a1d7539d5.png)

六、Head
------

### Loss

**经典 IoU loss**

IoU 算法是使用最广泛的算法，大部分的检测算法都是使用的这个算法。

![](https://img-blog.csdnimg.cn/791871aff5f74ada9e67bd2fc8bce3de.png)

**不足：**没有相交则 IOU=0 无法梯度计算，相同的 IOU 却反映不出实际情况

![](https://img-blog.csdnimg.cn/96446827681d4db69e36935cb3743698.png)

**GIOU（Generalized IoU）损失**

GIoU 考虑到，当检测框和真实框没有出现重叠的时候 IoU 的 loss 都是一样的，因此 GIoU 就引入了最小封闭形状 C（C 可以把 A，B 包含在内），在不重叠情况下能让预测框尽可能朝着真实框前进，这样就可以解决检测框和真实框没有重叠的问题。

![](https://img-blog.csdnimg.cn/79bde21e61384719b73eeb8c460db57e.png)

  
**公式：**![](https://img-blog.csdnimg.cn/08392acc0894442caa168e27560ddc7c.png)

**不足：**但是在两个预测框完全重叠的情况下，不能反映出实际情况

![](https://img-blog.csdnimg.cn/baec118c6961480d9a80b24b32950356.png)

**DIOU（Distance IoU）损失**

DIoU 考虑到 GIoU 的缺点，也是增加了 C 检测框，将真实框和预测框都包含了进来，但是 DIoU 计算的不是框之间的交并，而是计算的每个检测框之间的欧氏距离，这样就可以解决 GIoU 包含出现的问题。

![](https://img-blog.csdnimg.cn/44a2c17dd2464568b7dd413a3be05e91.png)

**公式：**其中分子计算预测框与真实框的中心点欧式距离 d 分母是能覆盖预测框与真实框的最小 BOX 的对角线长度 c

![](https://img-blog.csdnimg.cn/4c75feb220304bb296230cc872da7728.png)

 **CIOU（Complete IoU）损失**

CIoU 就是在 DIoU 的基础上增加了检测框尺度的 loss，增加了长和宽的 loss，这样预测框就会更加的符合真实框。

**公式：**损失函数必须考虑三个几何因素：重叠面积，中心点距离，长宽比 其中α可以当做权重参数

![](https://img-blog.csdnimg.cn/59121283aab9468c84465707beb1c120.png)

> **总结：**
> 
> *   **IOU_Loss：**主要考虑检测框和目标框重叠面积。
> *   **GIOU_Loss：**在 IOU 的基础上，解决边界框不重合时的问题。
> *   **DIOU_Loss：**在 IOU 和 GIOU 的基础上，考虑边界框中心点距离的信息。
> *   **CIOU_Loss：**在 DIOU 的基础上，考虑边界框宽高比的尺度信息。

### **NMS**

**DIOU-NMS**

DIOU-NMS 不仅考虑 IOU 的值，还考虑两个框的中心点的距离。如果两个框之间的 IOU 比较大，但是他们中心点之间的距离比较远，则会被认为是不同物体的检测框而不会被过滤掉。

**公式：** 不仅考虑了 IoU 的值, 还考虑了两个 Box 中心点之间的距离 其中 M 表示高置信度候选框，Bi 就是遍历各个框跟置信度高的重合情况

![](https://img-blog.csdnimg.cn/c14bb87c7bf54eff93a382ccd2ba65e0.png)

 **SOFT-NMS**

对于重合度较大的不是直接剔除，而是施加惩罚。

![](https://img-blog.csdnimg.cn/301f12032ecf4cf3ae165e3d826d70b3.png)

> 本文参考：
> 
> [想读懂 YOLOV4，你需要先了解下列技术 (一) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/139764729 "想读懂YOLOV4，你需要先了解下列技术(一) - 知乎 (zhihu.com)")
> 
>  [想读懂 YOLOV4，你需要先了解下列技术 (二) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/141533907 "想读懂YOLOV4，你需要先了解下列技术(二) - 知乎 (zhihu.com)")