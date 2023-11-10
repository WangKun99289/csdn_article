> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501)

![](https://img-blog.csdnimg.cn/604624f4e57b43e395a9b045cdb97a85.gif)

![](https://img-blog.csdnimg.cn/909f9900ccc04f78bef654b6c23bb213.jpeg)

前言
--

在 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 中网络结构采用 **yaml** 作为配置文件，之前我们也介绍过，YOLOv5 配置了 4 种不同大小的[网络模型](https://so.csdn.net/so/search?q=%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)，分别是 **YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x**，这几个模型的结构基本一样，**不同的是 depth_multiple 模型深度和 width_multiple 模型宽度这两个参数**。 就和我们买衣服的尺码大小排序一样，[YOLOv5s](https://so.csdn.net/so/search?q=YOLOv5s&spm=1001.2101.3001.7020) 网络是 YOLOv5 系列中深度最小，特征图的宽度最小的网络。其他的三种都是在此基础上不断加深，不断加宽。所以，这篇文章我们就以 **yolov5s.yaml** 为例来介绍。

![](https://img-blog.csdnimg.cn/36b30cfaa812499498f5fec5dcf9b2c1.png)

**yaml 这个文件在 models 文件夹下**，我们了解这个文件还是很重要的，如果未来我们想改进算法的网络结构，需要通过 yaml 这种形式定义模型结构，也就是说需要先修改该文件中的相