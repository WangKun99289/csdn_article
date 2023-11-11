> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/130043351?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/a47bd2c9554c41cbb9a6a901de5b17aa.gif) 

![](https://img-blog.csdnimg.cn/82e1fd3fc0b1472d9491f0d161697fa8.jpeg)

前言
--

在上一篇文章中我们介绍了如何划分数据集，划分好之后我们的前期准备工作就已经全部完成了，下面开始训练自己的数据集吧！

![](https://img-blog.csdnimg.cn/0f228db522b740ff80eafb1176147506.gif)

**前期回顾：**

[YOLOv5 入门实践（1）——手把手带你环境配置搭建](https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501 "YOLOv5入门实践（1）——手把手带你环境配置搭建")

[YOLOv5 入门实践（2）——手把手教你利用 labelimg 标注数据集](https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501 "YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集")

[YOLOv5 入门实践（3）——手把手教你划分自己的数据集](https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501 "YOLOv5入门实践（3）——手把手教你划分自己的数据集")

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀**本人 ******[YOLOv5 源码](https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020 "YOLOv5源码")******详解系列：**  

[YOLOv5 源码逐行超详细注释与解读（1）——项目目录结构解析](https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析")

[​​​​​​YOLOv5 源码逐行超详细注释与解读（2）——推理部分 detect.py](https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501 "​​​​​​YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py")

[YOLOv5 源码逐行超详细注释与解读（3）——训练部分 train.py](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py")

[YOLOv5 源码逐行超详细注释与解读（4）——验证部分 val（test）.py](https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py")

[YOLOv5 源码逐行超详细注释与解读（5）——配置文件 yolov5s.yaml](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml")

[YOLOv5 源码逐行超详细注释与解读（6）——网络结构（1）yolo.py](https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py")

[YOLOv5 源码逐行超详细注释与解读（7）——网络结构（2）common.py](https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py")

**目录**

[前言](#t2)

 [一、配置文件](#t3)

[1.1 修改数据集配置文件](#t4)

[1.2 修改模型配置文件](#t5)

[二、训练模型](#t6)

[三、性能评价指标](#t7)

![](https://img-blog.csdnimg.cn/18ac94bc86f4493bbaf6a6c1945c4768.gif)

 一、配置文件
-------

在训练前我们首先来配置文件，通过之前的学习（[YOLOv5 源码逐行超详细注释与解读（5）——配置文件 yolov5s.yaml](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml")），我们知道 YOLOv5 训练数据都是通过调用 **yaml 文件**里我们已经整理好的数据。在这里，我们首先需要修改两个 yaml 文件中的参数。**一个是 data 目录下的相应的 yaml 文件（数据集配置文件），一个是 model 目录文件下的相应的 yaml 文件（模型配置文件）**。

### 1.1 修改数据集配置文件

首先在 **data** 的目录下新建一个 **yaml** 文件，自定义命名（嫌麻烦的话可以直接复制 voc.yaml 文件，重命名然后在文件内直接修改。）。

![](https://img-blog.csdnimg.cn/0f8658373ea24cbeb55ce9f615ac1895.png)

然后修改文件内的路径和参数。

**train** 和 **val** 就是上一篇文章中通过 split 划分好的数据集文件（最好要填绝对路径，有时候由目录结构的问题会莫名奇妙的报错）

下面是两个参数，根据自己数据集的检测目标个数和名字来设定：

*   **nc:** 存放检测目标类别个数
*   **name：** 存放检测目标类别的名字（个数和 nc 对应）

这里我做的是火焰检测，所以目标类别只有一个。

![](https://img-blog.csdnimg.cn/df9863aec1d74d5ba36c1451a4638e32.png)

> 注意：也可以在 data 目录下的 **coco.yaml** 上修改自己的路径、类别数和类别名称。
> 
> 若在训练时报错，解决方法是：**冒号后面需要加空格，否则会被认为是字符串而不是字典。** 

### 1.2 修改模型配置文件

在 model 文件夹下有 4 种不同大小的网络模型，分别是 **YOLOv5s、YOLOv5m、YOLOv5l、YOLOv5x**，这几个模型的结构基本一样，**不同的是 depth_multiple 模型深度和 width_multiple 模型宽度这两个参数**。

我们本次使用的是 **yolov5s.pt** 这个预训练权重，同上修改 data 目录下的 yaml 文件一样，我们最好将 yolov5s.yaml 文件复制一份，然后将其重命名，我将其重命名为 yolov5_fire.yaml。

 ![](https://img-blog.csdnimg.cn/b76246ba72e7404a9a322901a6a59e31.png)

 同样，这里改一下 nc

![](https://img-blog.csdnimg.cn/691883a3eba540d3bc4ccd464f4993f5.png)

二、训练模型
------

训练模型是通过 **train.py** 文件，在训练前我们先介绍一下文件内的参数

```
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s_fire.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
 
    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
 
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
```

> **opt 参数解析：** 
> 
> *   **cfg:  ** 模型配置文件，网络结构
> *   **data:  ** 数据集配置文件，数据集路径，类名等
> *   **hyp:  ** 超参数文件
> *   **epochs:**   训练总轮次
> *   **batch-size:**   批次大小
> *   **img-size:**   输入图片分辨率大小
> *   **rect:**   是否采用矩形训练，默认 False
> *   **resume:**   接着打断训练上次的结果接着训练
> *   **nosave:**   不保存模型，默认 False
> *   **notest:**   不进行 test，默认 False
> *   **noautoanchor:**   不自动调整 anchor，默认 False
> *   **evolve:**   是否进行超参数进化，默认 False
> *   **bucket:**   谷歌云盘 bucket，一般不会用到
> *   **cache-images:**   是否提前缓存图片到内存，以加快训练速度，默认 False
> *   **weights:**   加载的权重文件
> *   **name:**   数据集名字，如果设置：results.txt to results_name.txt，默认无
> *   **device:**   训练的设备，cpu；0(表示一个 gpu 设备 cuda:0)；0,1,2,3(多个 gpu 设备)
> *   **multi-scale:**   是否进行多尺度训练，默认 False
> *   **single-cls:**    数据集是否只有一个类别，默认 False
> *   **adam:**   是否使用 adam 优化器
> *   **sync-bn:**   是否使用跨卡同步 BN, 在 DDP 模式使用
> *   **local_rank:**   gpu 编号
> *   **logdir:**   存放日志的目录
> *   **workers:** dataloader 的最大 worker 数量
> 
> （**train.py** 更多学习请看：[YOLOv5 源码逐行超详细注释与解读（3）——训练部分 train.py](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py")）

参数虽然很多，但是需要我们修改的很少：

![](https://img-blog.csdnimg.cn/aacdff03e6254156b0cb26d8a6666fbc.png)

```
parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s_fire.pt', help='initial weights path')
```

 **--weight ：**先选用官方的 yolov5s.pt 权重，当自己的训练完成后可更换为自己的权重

```
parser.add_argument('--cfg', type=str, default='models/yolov5s_fire.yaml', help='model.yaml path')
```

 **--cfg：**选用上一步 model 目录下我们刚才改好的模型配置文件

```
parser.add_argument('--data', type=str, default=ROOT / 'data/fire.yaml', help='dataset.yaml path')
```

 **--data：**选用上一步 data 目录下我们刚才改好的数据集配置文件

```
parser.add_argument('--epochs', type=int, default=300)
```

 **--epoch：**指的就是训练过程中整个数据集将被迭代多少轮，默认是 300，显卡不行就调小点

```
parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
```

**--batch-size：**一次看完多少张图片才进行权重更新，默认是 16，显卡不行就调小点

```
parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
```

 **--workers:** dataloader 的最大 worker 数量，一般用来处理多线程问题，默认是 8，显卡不行就调小点

以上都设置好了就可以开始训练啦~

![](https://img-blog.csdnimg.cn/8af234488e604ccb87544f3011af59c5.png)

若干个 hours 之后~ 

训练结果会保存在 **runs** 的 **train 文件**里。

![](https://img-blog.csdnimg.cn/4b4aec1656c54d1faf29d7cefa706f15.png)

至此，我们的训练就全部完成了~ 

 三、性能评价指标
---------

模型性能评价好坏主要看训练完成后得到的 results 图（先忽略我的精度），我们看看这里面都有啥：

![](https://img-blog.csdnimg.cn/28aacd4367f44a5da2f1c052d146002e.png)

*   **box_loss：** 推测为 GIoU 损失函数均值，越小方框越准；
*   **obj_loss：** 推测为目标检测 loss 均值，越小目标检测越准；
*   **cls_loss：** 推测为分类 loss 均值，越小分类越准；
*   **precision：** 准确率（找对的 / 找到的）；
*   **recall：** 召回率（找对的 / 该找对的）；
*   **mAP@0.5 & mAP@0.5:0.95：**  就是 mAP 是用 Precision 和 Recall 作为两轴作图后围成的面积，m 表示平均，@后面的数表示判定 iou 为正负样本的阈值，@0.5:0.95 表示阈值取 0.5:0.05:0.95 后取均值。（0.5 是 iou 阈值 = 0.5 时 mAP 的值），mAP 只是一个形容 PR 曲线面积的代替词叫做平均准确率，越高越好。

> 本文参考：
> 
>  [yolov5 训练相关参数解释](https://blog.csdn.net/weixin_41990671/article/details/107300314?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov5results%E5%9B%BE%E7%89%87%E5%8F%82%E6%95%B0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-107300314.142%5Ev82%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&spm=1018.2226.3001.4187 " yolov5训练相关参数解释")
> 
>  [目标检测 --- 教你利用 yolov5 训练自己的目标检测模型_](https://blog.csdn.net/didiaopao/article/details/119954291?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168100760816800197099112%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168100760816800197099112&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-2-119954291-null-null.142%5Ev82%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=yolov5%E8%AE%AD%E7%BB%83%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4187 "目标检测---教你利用yolov5训练自己的目标检测模型_")

![](https://img-blog.csdnimg.cn/79543f3cc75d438e93c233b569f7b20f.gif)