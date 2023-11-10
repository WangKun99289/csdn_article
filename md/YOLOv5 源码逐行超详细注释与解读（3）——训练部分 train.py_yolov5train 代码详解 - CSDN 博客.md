> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/fddc3f7753464670a5d79a1a7700cac9.gif)

![](https://img-blog.csdnimg.cn/cd78ca8c59b84c30b3109fd3c83bcc5e.jpeg)

前言
--

本篇文章主要是对 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 项目的训练部分 **train.py**。通常这个文件主要是用来读取用户自己的数据集，加载模型并训练。

文章代码逐行手打注释，每个模块都有对应讲解，一文帮你梳理整个代码逻辑！

**友情提示：**全文近 5 万字，可以先点![](https://img-blog.csdnimg.cn/ab1ecfc8b12a442b8f17b23e493963bc.gif)再慢慢看哦~

**源码下载地址：**[mirrors / ultralytics / yolov5 · GitCode](https://gitcode.net/mirrors/ultralytics/yolov5?utm_source=csdn_github_accelerator "mirrors / ultralytics / yolov5 · GitCode")