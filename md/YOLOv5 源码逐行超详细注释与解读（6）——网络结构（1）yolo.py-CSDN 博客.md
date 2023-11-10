> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/a737ca4082d54a16942eba727eae70c7.gif)

![](https://img-blog.csdnimg.cn/332c1ecb96b94f2d8dc5ed911ddbd0ef.jpeg)

前言
--

在上一篇中，我们简单介绍了 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 的配置文件之一 **yolov5s.yaml**，这个文件中涉及很多参数，它们的调用会在这篇 **yolo.py** 和下一篇 **common.py** 中具体实现。

本篇我们会介绍 **yolo.py**，这是 [YOLO](https://so.csdn.net/so/search?q=YOLO&spm=1001.2101.3001.7020) 的特定模块，和网络构建有关。**在 YOLOv5 源码中，模型的建立是依靠 yolo.py 中的函数和对象完成的**，这个文件主要由三个部分：**parse_model 函数**、**Detect 类**