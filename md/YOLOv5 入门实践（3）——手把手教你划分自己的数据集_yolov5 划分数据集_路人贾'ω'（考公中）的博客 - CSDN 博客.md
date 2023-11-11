> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/130025866?spm=1001.2014.3001.5501)

#### ![](https://img-blog.csdnimg.cn/1bf8a03fddd1469987acea9250895fd7.gif)

### ![](https://img-blog.csdnimg.cn/989a8ec219c4476ea1a9b4048128e5c6.jpeg)

前言
--

上一篇我们学习了如何利用 [labelimg](https://so.csdn.net/so/search?q=labelimg&spm=1001.2101.3001.7020) 标注自己的数据集，下一步就是该对这些数据集进行划分了。面对繁杂的数据集，如果手动划分的话不仅麻烦而且不能保证随机性。本篇文章就来手把手教你利用代码，自动将自己的数据集划分为训练集、验证集和测试集。一起来学习吧！

![](https://img-blog.csdnimg.cn/233127243a2d4ad0a367bf353465ae21.gif)

**前期回顾：**

 [YOLOv5 入门实践（1）——手把手带你环境配置搭建](https://blog.csdn.net/weixin_43334693/article/details/129981848?spm=1001.2014.3001.5501 "YOLOv5入门实践（1）——手把手带你环境配置搭建")

[YOLOv5 入门实践（2）——手把手教你利用 labelimg 标注数据集](https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501 "YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集")

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀**本人 [YOLOv5 源码](https://so.csdn.net/so/search?q=YOLOv5%E6%BA%90%E7%A0%81&spm=1001.2101.3001.7020 "YOLOv5源码")详解系列：**  

[YOLOv5 源码逐行超详细注释与解读（1）——项目目录结构解析](https://blog.csdn.net/weixin_43334693/article/details/129356033?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（1）——项目目录结构解析")

[​​​​​​YOLOv5 源码逐行超详细注释与解读（2）——推理部分 detect.py](https://blog.csdn.net/weixin_43334693/article/details/129349094?spm=1001.2014.3001.5501 "​​​​​​YOLOv5源码逐行超详细注释与解读（2）——推理部分detect.py")

[YOLOv5 源码逐行超详细注释与解读（3）——训练部分 train.py](https://blog.csdn.net/weixin_43334693/article/details/129460666?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（3）——训练部分train.py")

[YOLOv5 源码逐行超详细注释与解读（4）——验证部分 val（test）.py](https://blog.csdn.net/weixin_43334693/article/details/129649553?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（4）——验证部分val（test）.py")

[YOLOv5 源码逐行超详细注释与解读（5）——配置文件 yolov5s.yaml](https://blog.csdn.net/weixin_43334693/article/details/129697521?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（5）——配置文件yolov5s.yaml")

[YOLOv5 源码逐行超详细注释与解读（6）——网络结构（1）yolo.py](https://blog.csdn.net/weixin_43334693/article/details/129803802?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（6）——网络结构（1）yolo.py")

[YOLOv5 源码逐行超详细注释与解读（7）——网络结构（2）common.py](https://blog.csdn.net/weixin_43334693/article/details/129854764?spm=1001.2014.3001.5501 "YOLOv5源码逐行超详细注释与解读（7）——网络结构（2）common.py")

**目录**

[前言](#%E5%89%8D%E8%A8%80)

[一、训练集、测试集、验证集介绍](#%E4%B8%80%E3%80%81%E8%AE%AD%E7%BB%83%E9%9B%86%E3%80%81%E6%B5%8B%E8%AF%95%E9%9B%86%E3%80%81%E9%AA%8C%E8%AF%81%E9%9B%86%E4%BB%8B%E7%BB%8D)

[二、准备自己的数据集](#%E4%BA%8C%E3%80%81%E5%87%86%E5%A4%87%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86)

[三、划分的代码及讲解](#%C2%A0%E4%B8%89%E3%80%81%E5%88%92%E5%88%86%E7%9A%84%E4%BB%A3%E7%A0%81%E5%8F%8A%E8%AE%B2%E8%A7%A3)

![](https://img-blog.csdnimg.cn/58a2da161ad743d8b8735de5bd142014.gif)

一、训练集、测试集、[验证集](https://so.csdn.net/so/search?q=%E9%AA%8C%E8%AF%81%E9%9B%86&spm=1001.2101.3001.7020)介绍
------------------------------------------------------------------------------------------------------

我们通常把训练的数据分为三个文件夹：**训练集、测试集和验证集**。

我们来举一个栗子：模型的训练与学习，类似于老师教学生学知识的过程。

*   **1、训练集（train set）：**用于**训练模型以及确定参数**。相当于老师教学生知识的过程。
*   **2、验证集（validation set）：**用于**确定网络结构以及调整模型的超参数**。相当于月考等小测验，用于学生对学习的查漏补缺。
*   **3、测试集（test set）：**用于**检验模型的泛化能力**。相当于大考，上战场一样，真正的去检验学生的学习效果。

> **参数（parameters）：**指由模型通过学习得到的变量，如权重和偏置。

> **超参数（hyperparameters）：**指根据经验进行设定的参数，如迭代次数，隐层的层数，每层神经元的个数，学习率等。

![](https://img-blog.csdnimg.cn/c3c37c4064974a2e80dcf4562ae12c1e.png)

二、准备自己的数据集
----------

**第 1 步：在 YOLOv5 项目下创建对应文件夹**

在 YOLOv5 项目目录下创建 **datasets 文件夹**（名字自定义），接着在该文件夹下新建 **Annotations** 和 **images** **文件夹。**

*   **Annotations：**存放标注的标签文件
*   **images：**存放需要打标签的图片文件

如下图所示： 

![](https://img-blog.csdnimg.cn/d7e1c690a8e24881b7eb2fbf2729159f.png)

 **第 2 步：打开 labelimg 开始标注数据集**

使用教程可以看我的上一篇介绍： [YOLOv5 入门实践（2）——手把手教你利用 labelimg 标注数据集](https://blog.csdn.net/weixin_43334693/article/details/129995604?spm=1001.2014.3001.5501 " YOLOv5入门实践（2）——手把手教你利用labelimg标注数据集")

标注后 **Annotations 文件夹**下面为 xml 文件，如下图所示：

![](https://img-blog.csdnimg.cn/546910e49f3949a88ff751f9831f87c8.png)

 **images 文件夹**是我们的数据集图片，格式为 jpg，如下图所示： ![](https://img-blog.csdnimg.cn/24d94e63b33743f38de229c22405e26e.png)

**第 3 步：创建保存划分后数据集的文件夹**

创建一个名为 **ImageSets 的文件夹**（名字自定义），用来保存一会儿划分好的训练集、测试集和验证集。

![](https://img-blog.csdnimg.cn/9f2b4dafef2040ba84bea9abfe60b465.png)

>  **准备工作的注意事项：**
> 
> *   所有训练所需的图像存于一个目录，所有训练所需的标签存于一个目录。
> *   图像文件与标签文件都统一的格式。
> *   图像名与标签名一一对应。

三、划分的代码及讲解
----------

完成以上工作我们就可以来进行数据集的划分啦！

**第 1 步：创建 split.py** 

在 YOLOv5 项目目录下创建 split.py 项目。

![](https://img-blog.csdnimg.cn/43186ecef51c4280854e08583a036ce2.png)

**第 2 步：将数据集打乱顺序**

通过上面我们知道，数据集有 **images** 和 **Annotations** 这两个文件，我们需要把这两个文件绑定，然后将其打乱顺序。  
首先设置空列表，将 for 循环读取这两个文件的每个数据放入对应表中，再将这两个文件通过 **zip（）函数**绑定，计算总长度。

```
def split_data(file_path,xml_path, new_file_path, train_rate, val_rate, test_rate):
    each_class_image = []
    each_class_label = []
    for image in os.listdir(file_path):
        each_class_image.append(image)
    for label in os.listdir(xml_path):
        each_class_label.append(label)
    data=list(zip(each_class_image,each_class_label))
    total = len(each_class_image)
```

然后用 **random.shuffle（）函数**打乱顺序，再将两个列表解绑。

```
random.shuffle(data)
    each_class_image,each_class_label=zip(*data)
```

**第 3 步：按照确定好的比例将两个列表元素分割**

分别获取 **train、val、test** 这三个文件夹对应的图片和标签。

```
train_images = each_class_image[0:int(train_rate * total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_images = each_class_image[int((train_rate + val_rate) * total):]
    train_labels = each_class_label[0:int(train_rate * total)]
    val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_labels = each_class_label[int((train_rate + val_rate) * total):]
```

**第 4 步：在本地生成文件夹，将划分好的数据集分别保存**

接下来就是设置相应的路径保存格式，将图片和标签对应保存下来。

```
for image in train_images:
        print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in train_labels:
        print(label)
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
 
    for image in val_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in val_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
 
    for image in test_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in test_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'test' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
```

**第 5 步：设置路径并设置划分比例**

这里要设置的有三个：

*   **file_path：**图片所在位置，就是 image 文件夹
*   **xml_path：**标签所在位置，就是 Annotation 文件夹
*   **new_file_path：**划分后三个文件的保存位置，就是 ImageSets 文件夹

```
if __name__ == '__main__':
    file_path = "D:\yolov5-6.1\datasets\image"
    xml_path = "D:\yolov5-6.1\datasets\Annotation"
    new_file_path = "D:\yolov5-6.1\datasets\ImageSets"
    split_data(file_path,xml_path, new_file_path, train_rate=0.7, val_rate=0.1, test_rate=0.2)
```

最后一行是设置划分比例，这里的比例分配大家可以随便划分，我选取的是 7:1:2。

至此，我们的数据集就划分好了。

来运行一下看看效果吧：

![](https://img-blog.csdnimg.cn/236da729b2a1440f82deb473a2f67844.png)

![](https://img-blog.csdnimg.cn/40612253406c4c5e990765676ece6c22.png)

 我们可以看到，数据集图片和标签已经划分成了 train、val 和 test 三个文件夹。

![](https://img-blog.csdnimg.cn/87689455b1d746deb05353d49be7293f.png)![](https://img-blog.csdnimg.cn/a35f0ad9acdb48b19a2e70de3fd81d5c.png)![](https://img-blog.csdnimg.cn/56d4b0ee50e54211b9840fc8b83ee35e.png)

 比例也符合 7:1:2

 **split.py 完整代码**

```
import os
import shutil
import random
 
random.seed(0)
 
 
def split_data(file_path,xml_path, new_file_path, train_rate, val_rate, test_rate):
    each_class_image = []
    each_class_label = []
    for image in os.listdir(file_path):
        each_class_image.append(image)
    for label in os.listdir(xml_path):
        each_class_label.append(label)
    data=list(zip(each_class_image,each_class_label))
    total = len(each_class_image)
    random.shuffle(data)
    each_class_image,each_class_label=zip(*data)
    train_images = each_class_image[0:int(train_rate * total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_images = each_class_image[int((train_rate + val_rate) * total):]
    train_labels = each_class_label[0:int(train_rate * total)]
    val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_labels = each_class_label[int((train_rate + val_rate) * total):]
 
    for image in train_images:
        print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in train_labels:
        print(label)
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
 
    for image in val_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in val_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
 
    for image in test_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)
 
    for label in test_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'test' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)
 
 
if __name__ == '__main__':
    file_path = "D:\yolov5-6.1\datasets\image"
    xml_path = "D:\yolov5-6.1\datasets\Annotation"
    new_file_path = "D:\yolov5-6.1\datasets\ImageSets"
    split_data(file_path,xml_path, new_file_path, train_rate=0.7, val_rate=0.1, test_rate=0.2)
```

本文参考：

[三天玩转 yolo——数据集格式转化及训练集和验证集划分 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/558476071 "三天玩转yolo——数据集格式转化及训练集和验证集划分 - 知乎 (zhihu.com)")

[【yolov5】将标注好的数据集进行划分（附完整可运行 python 代码）](https://blog.csdn.net/freezing_00/article/details/129097738?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168091987916800192298379%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168091987916800192298379&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-129097738-null-null.142%5Ev82%5Ekoosearch_v1,201%5Ev4%5Eadd_ask,239%5Ev2%5Einsert_chatgpt&utm_term=yolo%E6%95%B0%E6%8D%AE%E9%9B%86%E5%88%92%E5%88%86&spm=1018.2226.3001.4187 "【yolov5】将标注好的数据集进行划分（附完整可运行python代码）")

![](https://img-blog.csdnimg.cn/fc7661f3761c4748b8d251f4863810a2.gif)