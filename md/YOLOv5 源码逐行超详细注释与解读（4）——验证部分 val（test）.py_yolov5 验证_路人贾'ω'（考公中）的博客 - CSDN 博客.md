> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501)

![](https://img-blog.csdnimg.cn/3d3b8afcacca45c4938021ddd3ce0cc9.gif)

![](https://img-blog.csdnimg.cn/20c45edcecfb4bbdb4c967735f0c8fb7.jpeg)

前言 
---

本篇文章主要是对 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 项目的验证部分。这个文件之前是叫 test.py，后来改为 **val.py**。

在之前我们已经学习了推理部分 **detect.py** 和训练部分 **train.py** 这两个，而我们今天要介绍的验证部分 **val.py** 这个文件主要是 **train.py** 每一轮训练结束后，**用 val.py 去验证当前模型的 mAP、混淆矩阵等指标以及各个超参数是否是最佳**， 不是最佳的话修改 **train.py** 里面的结构；确定是最佳了再用 **detect.py** 去泛化使用。

总结一下这三个文件的区别：