> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129854764)

#### ![](https://img-blog.csdnimg.cn/afc8f0dbb2e1443da0a50764581042d9.gif)

![](https://img-blog.csdnimg.cn/073d2590ad5c4482ba66b2d00c799955.jpeg)

前言 
---

上一篇我们一起学习了 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 的网络模型之一 **yolo.py**，它这是 [YOLO](https://so.csdn.net/so/search?q=YOLO&spm=1001.2101.3001.7020) 的特定模块，而今天要学习另一个和网络搭建有关的文件——**common.py**，这个文件存放着 YOLOv5 网络搭建常见的通用模块。如果我们需要修改某一模块，那么就需要修改这个文件中对应模块的定义。

学这篇的同时，搭配[【YOLO 系列】YOLOv5 超详细解读（网络详解）](https://blog.csdn.net/weixin_43334693/article/details/129312409?spm=1001.2014.3001.5502 "【YOLO系列】YOLOv5超详细解读（网络详解）")这篇算法详解效果更好噢~

**common.py 文件位置在./models/common.py**

![](https://img-blog.csdnimg.cn/44acf7d146d34c52870cccd483325c70.png)​