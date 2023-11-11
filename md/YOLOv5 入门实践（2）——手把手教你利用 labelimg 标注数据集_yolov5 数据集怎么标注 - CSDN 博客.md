> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/62942eeb1a874a708ccaa70e7dc6516b.gif) 

![](https://img-blog.csdnimg.cn/b197e3b63f8047c29ffdeaa7106ac08d.jpeg)
----------------------------------------------------------------------

前言
--

上一篇我们已经搭建好了 [YOLOv5](https://so.csdn.net/so/search?q=YOLOv5&spm=1001.2101.3001.7020) 的环境（直通车→[YOLOv5 入门实践（1）——手把手带你环境配置搭建](https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501 "YOLOv5入门实践（1）——手把手带你环境配置搭建")），现在就开始第二步利用 [labelimg](https://so.csdn.net/so/search?q=labelimg&spm=1001.2101.3001.7020) 标注数据集吧！

![](https://img-blog.csdnimg.cn/61bda3b60ebe4bb89eb559c2ef3c874a.gif)

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

[一、labelimg 工具介绍](#t3)

[二、 labelimg 的下载](#t4)

[三、labelimg 的安装](#t5)

[四、labelImg 的使用](#t6)

[4.1 准备工作](#t7) 

[4.2 标注前的设置](#t8)

[4.3 开始标注](#t9)

![](https://img-blog.csdnimg.cn/5d9bdf546bc1495e90f2b0ea7cee53d2.gif)

 一、labelimg 工具介绍
----------------

Labelimg 是一个图形图像注释工具。

它是用 Python 编写的，并使用 Qt 作为其图形界面。

注释以 PASCAL VOC 格式保存为 XML 文件，这是使用的 ImageNet 格式。此外，它还支持 YOLO 格式和 CreateML 格式。

二、 labelimg 的下载
---------------

labelimg 的下载有两种：

**法 1：**从官网下载→下载地址：[https://github.com/tzutalin/labelImg](https://github.com/tzutalin/labelImg "https://github.com/tzutalin/labelImg") 

![](https://img-blog.csdnimg.cn/602f897fe5b545f69eaa3b07f07c03a9.png)

**法 2：**如果你和我一样懒就直接网盘下载吧（感谢提供资源的大佬！） 

> 链接：https://pan.baidu.com/s/19GoT4Tb0Mco1STgprxAjPw?pwd=j666   
> 提取码：j666 

三、labelimg 的安装
--------------

**第 1 步：利用 cd 命令进入 labelimg 所在的文件夹**

```
d:

```

```
cd [自己的文件位置]

```

![](https://img-blog.csdnimg.cn/58a0f5c21ece452baae92747146132d2.png)

 **第 2 步：安装 pyqt，这里我安装的是 pyqt5**

```
conda install pyqt=5


```

![](https://img-blog.csdnimg.cn/e3f04033655a4221a3b6310f2efb90f9.png)

 安装完成就是下图这样：

![](https://img-blog.csdnimg.cn/bbfc82b41d8c4abd917fe378226eee57.png)

**第 3 步：安装完成后，执行命令**

```
pyrcc5 -o libs/resources.py resources.qrc


```

![](https://img-blog.csdnimg.cn/0aac3bcf2c154480b6c2c3d0f8e8945f.png)

 这个命令没有返回结果。

**第 4 步：打开 labelimg**

```
python labelImg.py


```

![](https://img-blog.csdnimg.cn/dda1fddfc24d4ddfaa8349f2e8739ab2.png)

 这样就打开了呢~

![](https://img-blog.csdnimg.cn/691acd207b0b4fb09863e9d990352f9a.png)

四、labelImg 的使用
--------------

### 4.1 准备工作 

**第 1 步：在 yolov5 目录下新建一个名为 VOCData 的文件夹**

（这个是约定俗成，不这么做也行）

![](https://img-blog.csdnimg.cn/d215d99d2f09474cb2a90616487c0f2b.png)

**第 2 步：在 VOCData 的文件夹内建立 Annotations 和 images** **文件夹**

*   **Annotations：**存放标注的标签文件
*   **images：**存放需要打标签的图片文件

![](https://img-blog.csdnimg.cn/d90c4140427844e2975b29a3de231451.png)

### **4.2 标注前的设置**

将要标注的图片放入 **images** **文件夹**内，运行软件前可以更改下要标注的类别。这里选了三个类别：花、猫猫和鱼。

![](https://img-blog.csdnimg.cn/4625940bba224ea58342f5921f806b64.png)

然后我们在 labelimg 的 data 文件下找到 **predefined_classes.txt** 这个 txt 文档，在里面输入自定义的类别名称，如下图所示：

![](https://img-blog.csdnimg.cn/b61a3a347bc44a7ab70cd427e85222bf.png)

### **4.3 开始标注**

标注前我们先认识一下功能键。如下图所示：

![](https://img-blog.csdnimg.cn/b5bb23f489bf48bb85aed5ed2b575d5d.png)

 还有 view 的一些功能键，如下图所示：

![](https://img-blog.csdnimg.cn/a5e9dd80bfac490c8d0762121f22b2a0.png)

常用快捷键如下：

> **A：** 切换到上一张图片
> 
> **D：** 切换到下一张图片
> 
> **W：**调出标注十字架
> 
> **del ：** 删除标注框框
> 
> **Ctrl+u：** 选择标注的图片文件夹
> 
> **Ctrl+r：** 选择标注好的 label 标签存在的文件夹

接下来打开图片，按住鼠标左键就可以标注了。

![](https://img-blog.csdnimg.cn/f055c46c672f4349975bd1051173e3fd.png)

点击鼠标右键还可以移动选框位置和调整大小。

![](https://img-blog.csdnimg.cn/a5fda63dfa374af78f21aace176381a4.png)

标签打完以后可以去 Annotations 文件下看到标签文件已经保存在这个目录下。

![](https://img-blog.csdnimg.cn/dcaaf71391b34287ac81e4ad4a9e8560.png)

好了，lambelimg 的使用就讲到这里啦~

> 本文参考：
> 
> [目标检测 --- 利用 labelimg 制作自己的深度学习目标检测数据集](https://blog.csdn.net/didiaopao/article/details/119808973?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168077368116800182799614%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168077368116800182799614&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-119808973-null-null.142%5Ev81%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=labelimg&spm=1018.2226.3001.4187 "目标检测---利用labelimg制作自己的深度学习目标检测数据集")

![](https://img-blog.csdnimg.cn/1c75fa0e59aa40cc824435ec870b24cb.gif)