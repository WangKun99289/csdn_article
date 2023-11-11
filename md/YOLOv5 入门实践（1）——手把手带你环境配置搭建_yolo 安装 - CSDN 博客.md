> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/6ef288a034584791af0c1ccf0cccad6a.gif) 

![](https://img-blog.csdnimg.cn/1c537274f8ad489b98a3e60d88f3d991.jpeg)
----------------------------------------------------------------------

前言
--

这两天我将 pycharm 社区版换成了专业版，也顺带着把环境从 CPU 改成了 GPU 版，本篇文章也就是我个人配置过程的一个简单记录，希望能够帮到大家啦~

![](https://img-blog.csdnimg.cn/5267fbcad7d841f384da3e5d4bd596c5.gif)

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀本人 [YOLOv5 源码](https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020 "YOLOv5源码")详解系列：  

[YOLOv5 源码逐行超详细注释与解读（1）——项目目录结构解析](https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析")

[​​​​​​YOLOv5 源码逐行超详细注释与解读（2）——推理部分 detect.py](https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501 "​​​​​​YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py")

[YOLOv5 源码逐行超详细注释与解读（3）——训练部分 train.py](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py")

[YOLOv5 源码逐行超详细注释与解读（4）——验证部分 val（test）.py](https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py")

[YOLOv5 源码逐行超详细注释与解读（5）——配置文件 yolov5s.yaml](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml")

[YOLOv5 源码逐行超详细注释与解读（6）——网络结构（1）yolo.py](https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py")

[YOLOv5 源码逐行超详细注释与解读（7）——网络结构（2）common.py](https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py")

**目录**
------

[前言](#%E5%89%8D%E8%A8%80)

[一、了解所需配置](#t3)

[1.1 CUDA](#t4)

[1.2 cuDNN](#t5)

[1.3 Anconda](#t6)

[1.4 pycharm](#t7)

[1.5 pytorch](#t8)

[二、安装 CUDA 和 cuDNN](#t9)

[2.1 CUDA 的下载与安装](#t10)

[2.2 cuDNN 的下载与安装](#t11)

[三、安装 Anaconda](#t12) 

[四、安装 pytorch](#t13) 

[五、配置 YOLOv5 环境](#t14)

[六、测试](#t15)

![](https://img-blog.csdnimg.cn/b145058acd8d456ab453fd67bc622f98.gif)

一、了解所需配置
--------

### 1.1 CUDA

2006 年，NVIDIA 公司发布了 CUDA(Compute Unified Device Architecture)，是一种新的操作 GPU 计算的硬件和软件架构，是建立在 NVIDIA 的 GPUs 上的一个通用并行计算平台和编程模型，它提供了 GPU 编程的简易接口，基于 CUDA 编程可以构建基于 GPU 计算的应用程序，利用 GPUs 的并行计算引擎来更加高效地解决比较复杂的计算难题。它将 GPU 视作一个数据并行计算设备，而且无需把这些计算映射到图形 API。操作系统的多任务机制可以同时管理 CUDA 访问 GPU 和图形程序的运行库，其计算特性支持利用 CUDA 直观地编写 GPU 核心程序。

### 1.2 [cuDNN](https://so.csdn.net/so/search?q=cuDNN&spm=1001.2101.3001.7020)

cuDNN 是 NVIDIACUDA® 深度神经网络库，是 GPU 加速的用于深度神经网络的原语库。cuDNN 为标准例程提供了高度优化的实现，例如向前和向后卷积，池化，规范化和激活层。  
全球的深度学习研究人员和框架开发人员都依赖 cuDNN 来实现高性能 GPU 加速。它使他们可以专注于训练神经网络和开发软件应用程序，而不必花时间在底层 GPU 性能调整上。cuDNN 的加快广泛使用的深度学习框架，包括 Caffe2，Chainer，Keras，MATLAB，MxNet，PyTorch 和 TensorFlow。已将 cuDNN 集成到框架中的 NVIDIA 优化深度学习框架容器，访问 NVIDIA GPU CLOUD 了解更多信息并开始使用。

### 1.3 Anconda

Anaconda 指的是一个开源的 Python 发行版本，其包含了 conda、Python 等 180 多个科学包及其依赖项。 因为包含了大量的科学包，Anaconda 的下载文件比较大（约 531 MB），如果只需要某些包，或者需要节省带宽或存储空间，也可以使用 Miniconda 这个较小的发行版（仅包含 conda 和 Python）。  
Anaconda 包括 Conda、Python 以及一大堆安装好的工具包，比如：numpy、pandas 等

### 1.4 pycharm

pycharm 是一个用于计算机编程的集成开发环境，主要用于 python 语言开发，并支持使用 Django 进行网页开发。简单来说就是人工智能的便捷语言。

### 1.5 pytorch

[PyTorch](https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020 "PyTorch") 是一个开源的 Python 机器学习库，其前身是 2002 年诞生于纽约大学 的 Torch。它是美国 Facebook 公司使用 python 语言开发的一个深度学习的框架，2017 年 1 月，Facebook 人工智能研究院（FAIR）在 GitHub 上开源了 PyTorch。

想进一步学习 pytorch 的友友，欢迎关注我的专栏噢！→[Pytorch_路人贾'ω'的博客 - CSDN 博客](https://blog.csdn.net/weixin_43334693/category_12186888.html?spm=1001.2014.3001.5482 "Pytorch_路人贾'ω'的博客-CSDN博客")

二、安装 CUDA 和 cuDNN
-----------------

官方教程

**CUDA：**[https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html "https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html")

**cuDNN：**[Installation Guide :: NVIDIA Deep Learning cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installwindows "Installation Guide :: NVIDIA Deep Learning cuDNN Documentation")

### 2.1 CUDA 的下载与安装

**1. 首先查看自己 CUDA 的版本**，有以下两种方法：

（1）打开 [nvidia](https://so.csdn.net/so/search?q=nvidia&spm=1001.2101.3001.7020 "nvidia")（桌面右键）-> 选择左下角的系统信息 -> 组件

![](https://img-blog.csdnimg.cn/e9b06bb5f9524d0cb990e3221142d521.png)

![](https://img-blog.csdnimg.cn/8982dce5452d4395b5c8087eca2868c7.png)

（2）直接在 cmd 中输入

```
nvidia-smi

```

这里就可以直接查看啦 

![](https://img-blog.csdnimg.cn/dae2eea30a54471fa1fde37061c6801e.png)  

 **2. 然后开始进入官网下载对应版本**，下载地址→  [官方驱动 | NVIDIA](https://www.nvidia.cn/Download/index.aspx?lang=cn "官方驱动 | NVIDIA")

![](https://img-blog.csdnimg.cn/df071599a7f2419383d08af35c79d5d2.png)

根据自己查到的版本下载对应既可。

然后就是漫长的等待 ing

![](https://img-blog.csdnimg.cn/fbcde3d1436d4c9097e38fadd6364f25.png)

 **3. 下载完了就开始安装**

![](https://img-blog.csdnimg.cn/c18232060f824fc191b7b8397be952d2.png)

 点击下一步

![](https://img-blog.csdnimg.cn/927e49349759432f8ec94a00e78c1a1a.png)

 这两个可以不用勾选

![](https://img-blog.csdnimg.cn/8b1bdfec8edd4ce286b5a085d2564fb8.png)

**4. 查看环境变量**

点击开始 --> 搜索高级系统设置 --> 查看环境变量

【如果没有需要自己添加】

![](https://img-blog.csdnimg.cn/0a053a8a259e4a75a249e12369161827.png)

 ![](https://img-blog.csdnimg.cn/c16e30a9b2a8406ab47e63ff9e364ef5.png)

 ![](https://img-blog.csdnimg.cn/8f8e878fc83b49dbbffec51b265183ae.png)

 一共四个系统变量，都是自动生成的，但是有时后两个系统变量可能不会自动生成，需要自己添加上去，添加时注意路径。

**5. 验证 CUDA 是否安装成功**

win+r，运行 cmd，输入

```
nvcc --version 
OR
nvcc -V
```

即可查看版本号

![](https://img-blog.csdnimg.cn/7d58e26204124fccb19a2448d1292a0c.png) 输入

```
set cuda

```

即可查看 CUDA 设置的环境变量 ![](https://img-blog.csdnimg.cn/18c321f76959404194d6ac746e2cbe3f.png)

 至此，CUDA 就已安装完成。但是在完成张量加速运算时还需要 cuDNN 的辅助，所以接下来我们来安装 cuDNN。

### 2.2 cuDNN 的下载与安装

**1. 查看与 CUDA 对应的 cuDNN 版本**

![](https://img-blog.csdnimg.cn/b0a1c52eb1ec41f8a38533b71d5a7955.png)

 **2. 在官网上下载。官网地址→**[cuDNN Download | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download "cuDNN Download | NVIDIA Developer")

点击注册

![](https://img-blog.csdnimg.cn/bf6706ea45e3491e8d8002c974d9770d.png)  注册成功

![](https://img-blog.csdnimg.cn/707898223fde491286f2257d22f473e6.png)

  **3. 开始下载**

**![](https://img-blog.csdnimg.cn/f3dee2b76d4649f29103f463c86a44df.png)**

 **4. 解压文件**

**![](https://img-blog.csdnimg.cn/7c828d1344794f74a51b674697d9a970.png)**

 我们下载后发现其实 cudnn 不是一个 exe 文件，而是一个压缩包，解压后，有三个文件夹，把三个文件夹拷贝到 cuda 的安装目录下。

![](https://img-blog.csdnimg.cn/bf68e36b930443089f5e9fb927664878.png) cuDNN 其实是 CUDA 的一个补丁，专为深度学习运算进行优化的。然后再添加环境变量

**5. 添加至系统变量**

往系统环境变量中的 path 添加如下路径（根据自己的路径进行修改） ![](https://img-blog.csdnimg.cn/126ad2fe10d94930bdc39100f90f3e86.png)

**7. 验证 cuDNN 是否安装成功** 

win+r，启动 cmd，cd 到安装目录下的.\extras\demo_suite，输入

```
原目录.\extras\demo_suite

```

然后分别输入.\bandwidthTest.exe 和.\deviceQuery.exe（进到目录后需要直接输 “bandwidthTest.exe” 和“deviceQuery.exe”）

```
.\bandwidthTest.exe

```

```
.\deviceQuery.exe

```

得到下图：

![](https://img-blog.csdnimg.cn/e68b46c944ac4fb8ad8159a3777bb4df.png)

 ![](https://img-blog.csdnimg.cn/2f7fba9831374786908ca8ef5e5d7f3a.png)

 至此，CUDA 和 cuDNN 已全部安装完毕~

三、**安装 Anaconda** 
------------------

因为我的电脑已经有了 Anaconda ，所以没有再安装。没有安装的可以看看这个教程：

[最新 Anaconda3 的安装配置及使用教程（详细过程）_HowieXue 的博客 - CSDN 博客](https://blog.csdn.net/HowieXue/article/details/118442904?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168076111016800226510971%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168076111016800226510971&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118442904-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=Anconda&spm=1018.2226.3001.4187 "最新Anaconda3的安装配置及使用教程（详细过程）_HowieXue的博客-CSDN博客")

四、安装 pytorch 
-------------

同样，pytorch 我电脑上也安装过了（不然咋出的专栏呢（手动狗头））。没有安装的推荐大家看我同门的这篇文章，步骤非常详细：[Win11 上 Pytorch 的安装并在 Pycharm 上调用 PyTorch 最新超详细_win11 安装 pytorch](https://blog.csdn.net/java1314777/article/details/128027977 "Win11上Pytorch的安装并在Pycharm上调用PyTorch最新超详细_win11安装pytorch")

五、配置 YOLOv5 环境
--------------

**1.yolov5 的源码下载**

**下载地址：**[mirrors / ultralytics / yolov5 · GitCode](https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator "mirrors / ultralytics / yolov5 · GitCode")

**方法一：git clone 到本地本地仓库**  
[指令]：`git clone https://github.com/ultralytics/yolov5`

**方法二：直接安装压缩包**  
没有安装 git 的话，可以直接点击 “克隆” 下载压缩包

 ![](https://img-blog.csdnimg.cn/c271d26de5d7445ea1a55341c0c8e63f.png)

**2. 预训练模型下载**

为了缩短网络的训练时间，并达到更好的精度，我们一般加载预训练权重进行网络的训练。  
YOLOv5 给我们提供了几个预训练权重，我们可以对应我们不同的需求选择不同的版本的预训练权重。在实际场景中是比较看这种速度，所以 YOLOv5s 是比较常用的。

将安装好的预训练模型放在 YOLO 文件下。

![](https://img-blog.csdnimg.cn/caa543a0d5704d9e96057c87d3bdd1f3.png)

  
**3. 安装 yolov5 的依赖项** 

首先创建虚拟环境并激活。conda 常用指令如下：

*   **创建虚拟环境：**
    
    ```
    conda create -n [虚拟环境名] python=[版本]
    
    ```
    

![](https://img-blog.csdnimg.cn/4018330236f14947a183ee59a3b0903c.png) 点 “y”

![](https://img-blog.csdnimg.cn/6bfb341e7aae42009a74d36c07e4b067.png)

*   **显示虚拟环境：**

```
conda env list

```

![](https://img-blog.csdnimg.cn/dd33f5676fb344edb7d70726740e85bb.png)

*    **激活虚拟环境：**

```
conda activate + [虚拟环境名]

```

![](https://img-blog.csdnimg.cn/f5763a1427864c8f8eb94f07229aa30a.png)

**4. 安装 pytorch-gup 版的环境**

由于 pytorch 的官网在国外，下载相关的环境包是比较慢的，所以我们给环境换源。在 pytorch 环境下执行如下的命名给环境换清华源。

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

这里要注意网速一定要好，不然就总报错。下图就是本人血泪史。

![](https://img-blog.csdnimg.cn/5c48c308979747d0bc7b9c323026815a.png)

 然后就安装完啦

![](https://img-blog.csdnimg.cn/9345406111d441748db4fd3e0a91a094.png)

 打开我们下载好的源码，点击设置 setting

![](https://img-blog.csdnimg.cn/619d95fe6b7140a6ba8bc6abaaaa68b7.png)

按以下步骤就 OK 啦！

![](https://img-blog.csdnimg.cn/323544fcca6a496c99bbad60c1ed5c6c.png)

六、测试
----

**1. 我们先运行 detect.py**

![](https://img-blog.csdnimg.cn/076e71f032ac4407bcb38e9fc9ae9f0f.png)

 这时会发现出现错误：

AttributeError: 'Upsample' object has no attribute 'recompute scale_factor'

![](https://img-blog.csdnimg.cn/78d72e1dc7824c9cb1ed97f4048ec2c1.png)

 **解决方法：**

点进蓝色的文件里下图对应位置，更改 forward 函数，复制一遍，去掉下面一行的代码

![](https://img-blog.csdnimg.cn/f686599e80eb4b6291d6472cd1ba99bf.jpeg)

再点击 run，结果就保存在 runs 的 detect 文件下了

![](https://img-blog.csdnimg.cn/c8ae9317d69c462bae2e359f10bcf00b.png) ![](https://img-blog.csdnimg.cn/3a2d3afbc36c45f9877c73c4d8c965d1.png)

**2. 我们再运行 train.py**

![](https://img-blog.csdnimg.cn/0640586c51d8498ab5e037503eef7bb5.png)

同样会发生报错

0SError: [winError 1455] 页面文件太小，无法完成操作。Error loading"D:\Anaconda3\envslyolov5-6.1lib\site-packages torch\lib\cudidTT"one of its dependencies

![](https://img-blog.csdnimg.cn/25e2c93a14d6473a8d3003b42945c99a.png)

 这就是因为我们 batchsize 和 workers 设置太大了的原因

 **解决方法：**

找到 train.py 的 parse_opt（）函数，将对应 batchsize 和 workers 参数调小，如下图：

![](https://img-blog.csdnimg.cn/94675c0382df4efcb260d0256742caa2.png)

 （你以为这样就完了吗？No！555~）

接着又会出现下面的错误：

OMP: Hint This means that multiple copies of the OpenMp runtime have been linked into theThat is dangerous, since it can degrade performance or cause incorrect results.. 

![](https://img-blog.csdnimg.cn/df8b83c3608245cbb4a3dc93e11337d3.png)

 **解决方法：**

在 import os 下面加入

```
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

```

 ![](https://img-blog.csdnimg.cn/d14b4e14ba3c487c8bfdf45722796c9d.png)

 （这下没错了吧？你想多了~）

然后又会出现这样的错误：

RuntimeError: resutt type float can't be cast to the desired output type ._int64 ![](https://img-blog.csdnimg.cn/298a28a749344de580f50c368e537c4d.png)

 **解决方法：**

首先进入 loss.py 文件，将 anchors = self.anchors[i] 改为

```
anchors, shape = self.anchors[i], p[i].shape

```

![](https://img-blog.csdnimg.cn/76d4db6a21ad4ebdb87750bbaf61fc3c.png)

**↓** ![](https://img-blog.csdnimg.cn/8d4118b53bed4a6ebc02bb464ddd8983.png)

 接着往下翻，将 indices.append((b, a, gj.clamp_(o, gain[3] - 1), gi.clamp_(0, gain[2] - 1)) 改为

```
indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))

```

![](https://img-blog.csdnimg.cn/bce300638f4d499badeeec320936e17f.png)

**↓** ![](https://img-blog.csdnimg.cn/fee952eb79ad4704bddd5e1d0c5e1354.png)

 到这终于能运行啦！撒花✿✿ヽ (°▽°) ノ✿

![](https://img-blog.csdnimg.cn/9748167f8ac649c1b5e980f2c8dfafe2.png)

 （训练时错误本来就有很多，但是错误原因网上都能找到的哦~）

到此为止，我们的环境就配好了。

本篇文章是我通过录屏复盘总结的，可能有一些地方有遗忘，大家要是配置过程中有问题还是要看看大佬们的教程（感谢大佬们！） 好了，我这个小白先撤了~ 下一篇再见啦！

> 本文参考：
> 
> [CUDA 与 cuDNN 安装教程（超详细）_kylinmin 的博客 - CSDN 博客](https://blog.csdn.net/anmin8888/article/details/127910084?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168074450216800192219938%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168074450216800192219938&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-4-127910084-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=CUDA&spm=1018.2226.3001.4187 "CUDA与cuDNN安装教程（超详细）_kylinmin的博客-CSDN博客")
> 
> [【零基础上手 yolov5】yolov5 的安装与相关环境的搭建_罅隙 ` 的博客 - CSDN 博客](https://blog.csdn.net/whc18858/article/details/127131741?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168068781616800217279306%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168068781616800217279306&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-5-127131741-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE&spm=1018.2226.3001.4187 "【零基础上手yolov5】yolov5的安装与相关环境的搭建_罅隙`的博客-CSDN博客  ")

![](https://img-blog.csdnimg.cn/55f80a8aec4f47c7bfbce91fd2ad2291.gif)