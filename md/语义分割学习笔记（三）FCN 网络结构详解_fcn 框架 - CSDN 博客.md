> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/qq_45981086/article/details/130436737?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%9C%B9%E9%9B%B3%E5%B7%B4%E6%8B%89wz%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0fcn&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~sobaiduweb~default-0-130436737.nonecase&spm=1018.2226.3001.4450)

推荐课程：[FCN 网络结构详解 (语义分割)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1J3411C7zd/?vd_source=a3264716fe097cbd43e5dbc235c0e426 "FCN网络结构详解(语义分割)_哔哩哔哩_bilibili")

感谢博主[霹雳吧啦 Wz ](https://space.bilibili.com/18161609 "霹雳吧啦Wz ") / [太阳花的小绿豆](https://blog.csdn.net/qq_37541097 "太阳花的小绿豆")提供视频讲解和源码支持，真乃神人也！

**目录**

[1.FCN 网络概述](#t0)

[2. 几种不同的 FCN 网络](#t1)

 [](#t2) [(1) FCN-32s](#t2)

 [](#t3) [(2) FCN-16s](#t3)

 [](#t4) [(3) FCN-8s](#t4)

[3. 损失计算](#t5)

#### 1.[FCN](https://so.csdn.net/so/search?q=FCN&spm=1001.2101.3001.7020) 网络概述

> FCN 网络（Fully Convolutional Networks）：首个端对端的针对像素级预测的全卷积网络。

![](https://img-blog.csdnimg.cn/21ef56f4e5e4430b88cc22b1c802ad32.png)

> FCN 网络思想：输入图像经过多次卷积，得到一个通道数为 21 的特征图，再经过上采样，得到一个与原图同样大小的特征图，再经过 Softmax 函数处理就能得到该像素针对 Pascal Voc 数据集每一个类别的预测概率，选择最大概率的类别作为该像素的预测类别。

FCN 网络在 [VGG](https://blog.csdn.net/qq_52358603/article/details/127762417 "VGG") 网络上做出的修改：把 **VGG 全连接层**改为卷积层。一方面，可以不用固定输入图像的大小。另一方面，当输入图像大小大于 24x24，最终得到的输出特征图的 channel 就会变为 2D 的数据，这时我们把 channel 提取出来就得到一张[热图（heatmap）](https://www.jianshu.com/p/398a20e78536 "热图（heatmap）")。

![](https://img-blog.csdnimg.cn/efc99928ffff4e1a874e314f1d20c7ca.png)

 最上面一个网络模型为 vgg 16。

#### 2. 几种不同的 FCN 网络

![](https://img-blog.csdnimg.cn/b5d4b3415aad4727988819a3f5f11324.png)

FCN-32s：使用 32 倍的上采样。FCN-16s：使用 16 倍的上采样。FCN-8s：使用 8 倍的上采样。

#### **(1) FCN-32s**

![](https://img-blog.csdnimg.cn/1d39d1b34e624a7993779c4b7437c127.png)

VGG16 Backbone（主干网络）为 VGG16 网络全连接层之前的网络部分。注意：FCN 网络把 **VGG 全连接层**改为卷积层，即其中两个卷积层为 FC6，FC7。

**模型的训练过程如下：**

1. 输入图片，首先，通过 VGG16 Backbone（主干网络）会将图片下采样 32 倍，得到的特征图 W、H 为原图片大小的 1/32，Channel 变为 512。

2. 其次，经过 size=7x7，padding=3，卷积核数为 4096 的 FC6 卷积，输出特征图大小不变，Channel 变为 4096。

3. 再次，经过 size=1x1，padding=1，卷积核数为 4096 为 FC7 卷积，输出特征图大小不变，Channel 也不变。

4. 然后，经过 size=1x1，padding=1，卷积核数为 num_class 的卷积，输出的特征图大小不变，Channel 变为 num_class。

（num_class 为分类个数，VGG 网络全连接层会经过 softmax 进行多分类，因此我们要把 Channel 值设置为分类个数，确保参数个数与 VGG 保持一致。讲解视频中有提到！）

5. 最后，经过一个 size=64 的上采样（即 32 倍的上采样），特征图恢复到原图大小。得到的特征图的 Channel 仍然为 num_class。

（在源码中，这里没有使用上采样，而是直接使用[双线性插值](https://blog.csdn.net/qq_37541097/article/details/112564822 "双线性插值")还原。原因是直接使用 32 倍的上采样效果不明显，不用也可以。这是由于直接放大 32 倍导致的。）

#### **(2) FCN-16s**

![](https://img-blog.csdnimg.cn/b672a74e52a64fa59f2a30f33ae6a8b1.png)

很明显 FCN-16s 网络在 VGG16 Backbone（主干网络）之后分为两个分支：

1. 最上面的分支其结构与 **FCN-32s** 的结构基本一致，唯一的不同在于**采用了 2 倍的上采样**（特征图大小扩大 2 倍），得到的特征图 size = 原图的 1/16，Channel=num_class。

2. 最下面的分支接受到 VGG16 主干网络中 MaxPool4 层输出的特征图（这里的特征图已经**经过了 16 倍的下采样**，大小为原图的 1/16），再经过 size=1x1，padding=1，卷积核数为 num_class 的卷积，得到 size = 原图的 1/16，Channel=num_class 的特征图。

3. 得到的两个特征图进行矩阵相加，得到一个新的特征图。

4. 最后，经过一个 16 倍的上采样，将特征图还原为原图大小。

#### **(3) FCN-8s**

![](https://img-blog.csdnimg.cn/36b65089ca804eeba769abfeb70f0a4f.png)

很明显 FCN-8s 网络一共有 3 条分支。自上而下命名为分支 1，分支 2，分支 3。

**模型的训练过程如下：**

1. 分支 1 和分支 2 整体的结构与 **FCN-16s** 基本一致，唯一的不同在两个特征图**相加后（第一个相加）**，**经过一个 2 倍的上采样**，得到一个 size = 原图大小的 1/8，Channel=num_class 的特征图。

2. 分支 3 接受到 VGG16 主干网络中 MaxPool3 层输出的特征图（这里的特征图已经**经过了 8 倍的下采样**，大小为原图的 1/8），再经过 size=1x1，padding=1，卷积核数为 num_class 的卷积，得到 size = 原图的 1/8，Channel=num_class 的特征图。

3. 得到的两个特征图进行矩阵相加，得到一个新的特征图。

4. 最后，经过一个 8 倍的上采样，将特征图还原为原图大小。

#### 3. 损失计算

![](https://img-blog.csdnimg.cn/ae2c58b41d424643992d9613e9eeaac1.png)

左边的通过训练模型最终得到的**特征图**，右边为**真实标记**。

**计算损失值过程：**

**1. 特征图**的每一个方格为一个 pixel（像素），如上图沿 Channel 方向每个 pixel 还有三个参数。沿 Channel 方向为每个像素做 **softmax 处理**，就能得到每个像素的**预测值**。将**预测值**与对应**真实值**（在真实标记对应位置）**计算交叉熵损失**。

**计算交叉熵损失公式：**

![](https://img-blog.csdnimg.cn/ae97bb0ef16d4b87a6311c732c0ece73.png)

2. 计算每一个像素的损失值，**求平均值**，最终得到整个网络模型的损失值。