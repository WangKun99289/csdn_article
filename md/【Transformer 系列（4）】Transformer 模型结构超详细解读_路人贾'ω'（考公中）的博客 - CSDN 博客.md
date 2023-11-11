> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/130250571)

![](https://img-blog.csdnimg.cn/61fd1fa0d8f343c284eecf24ba0dce48.jpeg)

前言 
---

前一篇我们一起读了 [Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020) 的论文《Attention Is All You Need》，不知道大家是否真的理解这个传说中的神（反正俺是没有~）

这两天我又看了一些视频讲解，感谢各位大佬的解读，让我通透了不少。

这篇文章就和大家分享一下我的理解！

![](https://img-blog.csdnimg.cn/62fb84afcd6d499e827f103a18686fb3.gif)

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀**前期回顾**

 [【Transformer 系列（1）】encoder（编码器）和 decoder（解码器）](https://blog.csdn.net/weixin_43334693/article/details/130161845?spm=1001.2014.3001.5502 "【Transformer系列（1）】encoder（编码器）和decoder（解码器）")

 [【Transformer 系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解](https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5502 " 【Transformer系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解")  
 [【Transformer 系列（3）】 《Attention Is All You Need》论文超详细解读（翻译＋精读）](https://blog.csdn.net/weixin_43334693/article/details/130208816?spm=1001.2014.3001.5502 "【Transformer系列（3）】 《Attention Is All You Need》论文超详细解读（翻译＋精读）")

**目录**

[前言](#t1) 

[🌟一、Transformer 整体结构](#t2) 

[🌟二、编码器：Encoder](#t3)

 [2.1 输入](#t4)

[2.1.1 词嵌入：Word Embedding 层](#t5)

[2.1.2 位置编码器：Position Embedding 层](#t6)

 [2.2 注意力机制](#t7)

[2.2.1 自注意力机制：Self-Attention](#t8)

[2.2.2 多头注意力机制：Multi-Head Attention](#t9)

[2.3 残差连接](#t10)

[2.4LN 和 BN](#t11)

[2.5 前馈神经网络：FeedForward](#t12)

[🌟三、解码器：Decoder](#t13)

 [3.1 第一个 Multi-Head Attention](#t14)

[3.1.1 掩码：Mask](#t15) 

[3.1.2 具体实现步骤](#t16)

 [3.2 第二个 Multi-Head Attention](#t17)

 [3.3Linear 和 softmax](#t18)

![](https://img-blog.csdnimg.cn/e27d5e1bc72f4e8587dc3349ff46cf86.gif)🌟一、Transformer 整体结构 
------------------------------------------------------------------------------------------

 首先我们回顾一下这个神图：

![](https://img-blog.csdnimg.cn/6f8173b5b7b64f678b3aa55dedf75b65.png)

 这张图小白刚看时会觉得很复杂有木有？其实 Transformer 主要就做了这件事：

![](https://img-blog.csdnimg.cn/927e0df3f2ee4f049d8fd91d2bcbe4d7.png)

可以看到 Transformer 由 **Encoder** 和 **Decoder** 两个部分组成，**Encoder 把输入读进去**，**Decoder 得到输出**： 

![](https://img-blog.csdnimg.cn/7b19edcbdb9b4b858f36c4cb189f14bf.png)

**Encoder** 和 **Decoder** 都包含 6 个 block。这 6 个 block 结构相同，但参数各自随机初始化。（

Encoder 和 [Decoder](https://so.csdn.net/so/search?q=Decoder&spm=1001.2101.3001.7020) 不一定是 6 层，几层都可以，原论文里采用的是 6 层。）

![](https://img-blog.csdnimg.cn/2db443b2830142c098ed03b3a00641ec.png)

🌟二、[编码器](https://so.csdn.net/so/search?q=%E7%BC%96%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)：Encoder
------------------------------------------------------------------------------------------------------

![](https://img-blog.csdnimg.cn/1c8c3afebedb4fd485c8711415a0be77.png)

    Encoder 由三个部分组成：**输入**、**多头注意力**、**前馈神经网络**。

###  2.1 输入

Transformer 中单词的输入表示 **x** 由 **Word Embedding** 和 **Position Embedding** 相加得到。

![](https://img-blog.csdnimg.cn/e28bc3235b174b50a62741e637f016a9.png)

####  2.1.1 词嵌入：Word Embedding 层

**词嵌入层** 负责将自然语言转化为与其对应的独一无二的词向量表达。将词汇表示成特征向量的方法有多种：

**（1）One-hot 编码**

 One-hot 编码使用一种常用的离散化特征表示方法，在用于词汇向量表示时，向量的列数为所有单词的数量，只有对应的词汇索引为 1，其余都为 0。

举个栗子，“我爱我的祖国” 这句话，总长为 6，但只有 5 个不重复字符，用 One-hot 表示后为 6*5 的矩阵，如图所示：

![](https://img-blog.csdnimg.cn/img_convert/52e56a1904d501394e39147fa9275755.png)

但是这种数据类型十分稀疏，即便使用很高的学习率，依然不能得到良好的学习效果。

**（2）数字表示**

数字表示是指用整个文本库中出现的词汇构建词典，以词汇在词典中的索引来表示词汇。所以与其叫做 “数字表示”，还不如叫 “索引表示”。

举个栗子，还是 “我爱我的祖国” 这句话，就是我们整个语料库，那么整个语料库有 5 个字符，假设我们构建词典 {'我':0, '爱':1, '的':2, '祖':3, '':4}，“我爱我的祖国” 这句话可以转化为向量：[0, 1, 0, 2, 3, 4]。如图所示。这种方式存在的问题就是词汇只是用一个单纯且独立的数字表示，难以表达出词汇丰富的语义。

![](https://img-blog.csdnimg.cn/img_convert/464db105cc77cf249f615d5d89eee03a.png)

#### 2.1.2 位置编码器：Position Embedding 层

Transformer 中除了 Word Embedding，还需要使用 Position Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，因此是无法捕捉到序列顺序信息的**，例如将 K、V 按行进行打乱，那么 Attention 之后的结果是一样的。但是序列信息非常重要，代表着全局的结构，因此必须将序列的分词相对或者绝对 position 信息利用起来。

Position Embedding 用 **PE** 表示，**PE** 的维度与 Word Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：

![](https://img-blog.csdnimg.cn/0824a9e8d0fd4b18a08d540c86282de1.png)

其中 pos 表示 positionindex， i 表示 dimension index。 

###  2.2 注意力机制

![](https://img-blog.csdnimg.cn/017599f36e974ada9d88d6ad1a899b7e.jpeg)

我们再来看一下这个图，图中红色圈中的部分为 **Multi-Head Attention**，是由多个 **Self-Attention** 组成的，可以看到 **Encoder** 包含一个 **Multi-Head Attention**，而 **Decoder** 包含两个 Multi-Head Attention (其中有一个用到 Masked)。

**Multi-Head Attention** 上方还包括一个 Add & Norm 层：

*   **Add：** 表示残差连接 (Residual Connection) 用于防止网络退化
*   **Norm：** 表示 Layer Normalization，用于对每一层的激活值进行归一化

#### 2.2.1 自注意力机制：Self-Attention

**自注意力机制**是注意力机制的变体，其减少了对外部信息的依赖，更擅长捕捉数据或特征的内部相关性。自注意力机制的关键点在于，Q、K、V 是同一个东西，或者三者来源于同一个 X，三者同源。通过 X 找到 X 里面的关键点，从而更关注 X 的关键信息，忽略 X 的不重要信息。不是输入语句和输出语句之间的注意力机制，**而是输入语句内部元素之间或者输出语句内部元素之间发生的注意力机制。**

**如何运用自注意力机制？** 

**第 1 步：得到 Q，K，V 的值**

对于每一个向量 x，分别乘上三个系数 ![](https://latex.csdn.net/eq?W%5E%7Bq%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWE3RnhENkFEVEwyb3I0S3NUYUVxcXJaWWpqdlNXeUNhM0ZVdFJ5SGVIaWNuUUV6bHFpYVBPbjRmVGNxMm5tNFc2M2IvNjQw?x-oss-process=image/format,png)， ![](https://latex.csdn.net/eq?W%5E%7Bk%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWExbWliWVVNeVROeDhWbm5VRlVibGJwZnNLWXl4T0ZvZ252NEI2U21HcUZ6ZHdKeW1PMmtveG9KR3l1aWFScU9JSzAvNjQw?x-oss-process=image/format,png)，![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWFEYmJ4d21YNnJJeU83T25QZ3laU2ZZRldCTzlTSjgyS1hEcmljdGJpYU15WUE2M0xUUllLdWdPUzNYejEwYnMzRzYvNjQw?x-oss-process=image/format,png)![](https://latex.csdn.net/eq?W%5E%7Bv%7D)，得到的 Q，K 和 V 分别表示 query，key 和 value

![](https://img-blog.csdnimg.cn/49fa756cd86b4d248baf53f91dec0521.jpeg)

【注意】三个 W 就是我们需要学习的参数。

**第 2 步：Matmul**

利用得到的 Q 和 K 计算每两个输入向量之间的相关性，一般采用点积计算，为每个向量计算一个 score：score =q · k 

![](https://img-blog.csdnimg.cn/9d4d3104ba0d4c82adb6662d74ef7234.jpeg)

**第 3 步：Scale+Softmax**

将刚得到的相似度除以![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bk%7D%7D)，再进行 Softmax。经过 Softmax 的归一化后，每个值是一个大于 0 且小于 1 的权重系数，且总和为 0，这个结果可以被理解成一个权重矩阵。

![](https://img-blog.csdnimg.cn/61cdd0e854ac45ef9314b962dc266e83.jpeg)

 **第 4 步：Matmul**

使用刚得到的权重矩阵，与 V 相乘，计算加权求和。

![](https://img-blog.csdnimg.cn/9c6b21bb37334c8bbf1c6ee89d4b5ef9.jpeg)

以上是对 **Thinking Machines** 这句话进行自注意力的全过程，最终得到 **z1** 和 **z2** 两个新向量。

其中 **z1** 表示的是 **thinking** 这个词向量的新的向量表示（通过 **thinking** 这个词向量，去查询和 **thinking machine** 这句话里面每个单词和 **thinking** 之间的相似度）。

也就是说新的 **z1** 依然是 **thinking** 的词向量表示，只不过这个词向量的表示蕴含了 **thinking** **machines** 这句话对于 **thinking** 而言哪个更重要的信息 。

#### 2.2.2 多头注意力机制：Multi-Head Attention

与其只使用单独一个注意力汇聚， 我们可以用独立学习得到的 **h** 组（一般 **h=8**）不同的线性投影来变换 Q、K 和 V。

然后，这 h 组变换后的 Q、K 和 V 将并行地送到注意力汇聚中。 最后，将这 h 个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换， 以产生最终输出。 这种设计被称为**多头注意力（multihead attention）**。

![](https://img-blog.csdnimg.cn/0e537c796e9745959e3c92bb3a0f7327.png)

**如何运用多头注意力机制？** 

**第 1 步：定义多组 W，生成多组 Q、K、V**

刚才我们已经理解了，Q、K、V 是输入向量 X 分别乘上三个系数 ![](https://latex.csdn.net/eq?W%5E%7Bq%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWE3RnhENkFEVEwyb3I0S3NUYUVxcXJaWWpqdlNXeUNhM0ZVdFJ5SGVIaWNuUUV6bHFpYVBPbjRmVGNxMm5tNFc2M2IvNjQw?x-oss-process=image/format,png)， ![](https://latex.csdn.net/eq?W%5E%7Bk%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWExbWliWVVNeVROeDhWbm5VRlVibGJwZnNLWXl4T0ZvZ252NEI2U21HcUZ6ZHdKeW1PMmtveG9KR3l1aWFScU9JSzAvNjQw?x-oss-process=image/format,png)，![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWFEYmJ4d21YNnJJeU83T25QZ3laU2ZZRldCTzlTSjgyS1hEcmljdGJpYU15WUE2M0xUUllLdWdPUzNYejEwYnMzRzYvNjQw?x-oss-process=image/format,png)![](https://latex.csdn.net/eq?W%5E%7Bv%7D)分别相乘得到的，  ![](https://latex.csdn.net/eq?W%5E%7Bq%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWE3RnhENkFEVEwyb3I0S3NUYUVxcXJaWWpqdlNXeUNhM0ZVdFJ5SGVIaWNuUUV6bHFpYVBPbjRmVGNxMm5tNFc2M2IvNjQw?x-oss-process=image/format,png)， ![](https://latex.csdn.net/eq?W%5E%7Bk%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWExbWliWVVNeVROeDhWbm5VRlVibGJwZnNLWXl4T0ZvZ252NEI2U21HcUZ6ZHdKeW1PMmtveG9KR3l1aWFScU9JSzAvNjQw?x-oss-process=image/format,png)，![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWFEYmJ4d21YNnJJeU83T25QZ3laU2ZZRldCTzlTSjgyS1hEcmljdGJpYU15WUE2M0xUUllLdWdPUzNYejEwYnMzRzYvNjQw?x-oss-process=image/format,png)![](https://latex.csdn.net/eq?W%5E%7Bv%7D)是可训练的参数矩阵。

现在，对于同样的输入 X，我们定义多组不同的 ![](https://latex.csdn.net/eq?W%5E%7Bq%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWE3RnhENkFEVEwyb3I0S3NUYUVxcXJaWWpqdlNXeUNhM0ZVdFJ5SGVIaWNuUUV6bHFpYVBPbjRmVGNxMm5tNFc2M2IvNjQw?x-oss-process=image/format,png)， ![](https://latex.csdn.net/eq?W%5E%7Bk%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWExbWliWVVNeVROeDhWbm5VRlVibGJwZnNLWXl4T0ZvZ252NEI2U21HcUZ6ZHdKeW1PMmtveG9KR3l1aWFScU9JSzAvNjQw?x-oss-process=image/format,png)，![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWFEYmJ4d21YNnJJeU83T25QZ3laU2ZZRldCTzlTSjgyS1hEcmljdGJpYU15WUE2M0xUUllLdWdPUzNYejEwYnMzRzYvNjQw?x-oss-process=image/format,png)![](https://latex.csdn.net/eq?W%5E%7Bv%7D) ，比如![](https://latex.csdn.net/eq?W_%7B0%7D%5E%7B%5E%7Bq%7D%7D)、![](https://latex.csdn.net/eq?W_%7B0%7D%5E%7B%5E%7Bk%7D%7D)、![](https://latex.csdn.net/eq?W_%7B0%7D%5E%7B%5E%7Bv%7D%7D)，![](https://latex.csdn.net/eq?W_%7B1%7D%5E%7B%5E%7Bq%7D%7D)、![](https://latex.csdn.net/eq?W_%7B1%7D%5E%7B%5E%7Bk%7D%7D)、![](https://latex.csdn.net/eq?W_%7B1%7D%5E%7B%5E%7Bv%7D%7D)每组分别计算生成不同的 Q、K、V，最后学习到不同的参数。

![](https://img-blog.csdnimg.cn/6e9d6d0b7a2d4f0c9121d26dedf01b5d.jpeg)

**第 2 步：定义 8 组参数**

对应 8 组  ![](https://latex.csdn.net/eq?W%5E%7Bq%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWE3RnhENkFEVEwyb3I0S3NUYUVxcXJaWWpqdlNXeUNhM0ZVdFJ5SGVIaWNuUUV6bHFpYVBPbjRmVGNxMm5tNFc2M2IvNjQw?x-oss-process=image/format,png)， ![](https://latex.csdn.net/eq?W%5E%7Bk%7D)![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWExbWliWVVNeVROeDhWbm5VRlVibGJwZnNLWXl4T0ZvZ252NEI2U21HcUZ6ZHdKeW1PMmtveG9KR3l1aWFScU9JSzAvNjQw?x-oss-process=image/format,png)，![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3N2Zy9XbXdxanNTQnNaSXFZTDkxUG4wS1NmZnBKOFViZloyaWFEYmJ4d21YNnJJeU83T25QZ3laU2ZZRldCTzlTSjgyS1hEcmljdGJpYU15WUE2M0xUUllLdWdPUzNYejEwYnMzRzYvNjQw?x-oss-process=image/format,png)![](https://latex.csdn.net/eq?W%5E%7Bv%7D) ，再分别进行 self-attention，就得到了![](https://latex.csdn.net/eq?Z_%7B0%7D) -![](https://latex.csdn.net/eq?Z_%7B7%7D)

![](https://img-blog.csdnimg.cn/238fda6a68af4151b9cdde8224b4612a.jpeg)

**第 3 步：将多组输出拼接后乘以矩阵![](https://latex.csdn.net/eq?W_%7B0%7D)以降低维度**

 首先在输出到下一层前，我们需要将![](https://latex.csdn.net/eq?Z_%7B0%7D) -![](https://latex.csdn.net/eq?Z_%7B7%7D)concat 到一起，乘以矩阵**![](https://latex.csdn.net/eq?W_%7B0%7D)**做一次线性变换降维，得到 Z。

![](https://img-blog.csdnimg.cn/3d14d6a9722848a186bd336d01524ff3.jpeg)

 **完整流程图如下：（感谢翻译的大佬！）**

![](https://img-blog.csdnimg.cn/f11c9a296d3a4a3ca365324f8a604564.png)

【注意】对于上图中的第 2）步，当前为第一层时，直接对输入词进行编码，生成词向量 X；当前为后续层时，直接使用上一层输出。 

### 2.3 残差连接

每个编码器的每个子层（Self-Attention 层和 FFN 层）都有一个残差连接，再执行一个层标准化操作。

![](https://img-blog.csdnimg.cn/6b74fad04bf24546928ef127e5f3ea8e.png)

把得到的两个词的 Attention 值摞在一起后，将 “加入位置编码后的词向量 **X**” 与 “摞在一起后输出的 Attention 值 **Z**” 相加。残差连接减小了梯度消失的影响。加入残差连接，就能保证层次很深的模型不会出现梯度消失的现象。

### 2.4LN 和 BN

*   **LN：**Layer Normalization，LN 是 “横” 着来的，对同一个样本，不同的特征做归一化。
*   **BN：**Batch Normalization，BN 是 “竖” 着来的，对不同样本，同一特征做归一化。

**二者提出的目的都是为了加快模型收敛，减少训练时间。**

![](https://img-blog.csdnimg.cn/img_convert/09ceec8dd6009d61f61519691bd68d51.jpeg)

【注意】在 NLP 任务中，一般选用的都是 LN，不用 BN。因为句子长短不一，每个样本的特征数很可能不同，造成很多句子无法对齐，所以不适合用 BN。

###  2.5 **前馈神经网络：FeedForward**

在进行了 Attention 操作之后，Encoder 和 Decoder 中的每一层都包含了**一个全连接前向网络**，对每个 position 的向量分别进行相同的操作，包括**两个线性变换和一个 ReLU 激活输出**： 

![](https://img-blog.csdnimg.cn/c037a46d1b54463f9b9f831e9c1b17f8.png)

假设多头注意力部分有两个头，那么输出的两个注意力头 Zi 分别通过两个 Feed Forward，然后接一个残差连接，即 **Zi 和 Feed Forward 的输出 Add 对位相加。**最后把相加的结果进行一次 LN 标准化。  
![](https://img-blog.csdnimg.cn/c3a25e667b0e4edf8575ce3e9cc86ce0.png)

🌟**三、解码器：**Decoder
-------------------

![](https://img-blog.csdnimg.cn/78942384545e4b5091f8df3d299ac460.png)

上图红色框框为 Transformer 的 **Decoder** 结构，与 **Encoder** 相似，但是存在一些区别。

**Decoder** 包含两个 **Multi-Head Attention** 层。

*   第一个 **Multi-Head Attention** 层采用了 **Masked** 操作。
*   第二个 **Multi-Head Attention** 层的 **K, V 矩阵**使用 **Encoder** 的**编码信息矩阵 C** 进行计算，而 **Q** 使用上一个 **Decoder** 的输出计算。
*   最后有一个 Softmax 层计算下一个翻译单词的概率。

###  **3.1 第一个 Multi-Head Attention**

#### ![](https://img-blog.csdnimg.cn/c86fc9844ca74465bd28b465a55cf579.png)

#### 3.1.1 掩码：Mask 

**Mask** 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 Padding Mask 和 Sequence Mask。其中，Padding Mask 在所有的 scaled dot-product attention 里面都需要用到，而 Sequence Mask 只有在 Decoder 的 Self-Attention 里面用到。

> 为什么需要 Mask？   
> 
>    有一些生成的 attention 张量中的值计算有可能已知了未来信息而得到的，未来信息被看到是因为训练时会把整个输出结果都一次性进行 Embedding，但是理论上解码器的的输出却不是一次就能产生最终结果的，而是一次次通过上一次结果综合得出的，因此，未来的信息可能被提前利用。所以，**Attention 中需要使用掩码张量掩盖未来信息**。
> 
>   我们可以这么来理解 Mask 的作用：我们建模目的就是为了达到预测的效果，**所谓预测，就是利用过去的信息 (此前的序列张量) 对未来的状态进行推断，如果把未来需要进行推断的结果，共同用于推断未来，那叫抄袭，不是预测。**当然，这么做的话训练时模型的表现会很好，但是在测试 (test) 时，模型表现会很差。

换句话说，我们是用一句话中的前 N − 1 个字预测第 N 个字，那么我们在预测第 N 个字时，就不能让模型看到第 N 个字之后的信息，所以这里们把预测第 N 个字时，第 N 包括) 个字之后的字都 Masked 掉。

我们来举个栗子：

![](https://img-blog.csdnimg.cn/1cf36b1634e043f088b11c5ed8bf5fae.png)![](https://img-blog.csdnimg.cn/3ac35d216305452d8e3dc93799c02486.png) 

如果像 **Encoder** 的注意力机制那里一样没有 Mask，那么在训练 **Decoder** 时，如果要生成预测结果 you，就需要用到下面整个句子的所有词（s,I,Love,You,Now）。但是在真正预测的时候，并看不到未来的信息（即还轮不到 You 和 Now 呢）。

所以在预测阶段，预测的第一步生成第一个词 **I** 的时候，用起始词 **<start>** 做 **self-attention**；然后预测的第二步生成第二个词 **Love** 的时候，就做 **<start>** 和 **I** 两个词的 **self-attention，**后面的词被掩盖了。以此往复，预测的每一步在该层都有一个输出 Q，Q 要送入到中间的 Multi-Head Attention 层，和 encoder 部分输出的 K，V 做 attention。

#### 3.1.2 具体实现步骤

**第一步：**是 Decoder 的输入矩阵和 **Mask** 矩阵，输入矩阵包含 "<Start> I Love You Now" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

![](https://img-blog.csdnimg.cn/img_convert/2a58698758d46ace4c4783ce6053fbd3.png) （输入矩阵与 Mask 矩阵）

**第二步：**接下来的操作和之前的 Self-Attention 一样，通过输入矩阵 **X** 计算得到 **Q,K,V** 矩阵。然后计算 **Q** 和![](https://latex.csdn.net/eq?K%5E%7BT%7D)的乘积![](https://latex.csdn.net/eq?QK%5E%7BT%7D)。

![](https://img-blog.csdnimg.cn/img_convert/747c5fa4ad071c14bcc15423470ffaaa.webp?x-oss-process=image/format,png) （Q 乘以 K 的转置）

**第三步：**在得到 ![](https://latex.csdn.net/eq?QK%5E%7BT%7D)之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用 **Mask** 矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![](https://img-blog.csdnimg.cn/img_convert/69889c62ac2e6600a9f1d6b113d06fee.webp?x-oss-process=image/format,png) （Softmax 之前 Mask）

得到 **Mask** ![](https://latex.csdn.net/eq?QK%5E%7BT%7D)之后在 **Mask** ![](https://latex.csdn.net/eq?QK%5E%7BT%7D)上进行 Softmax，每一行的和都为 1。但是单词 0 在单词 1, 2, 3, 4 上的 attention score 都为 0。

**第四步：**使用 **Mask** ![](https://latex.csdn.net/eq?QK%5E%7BT%7D)与矩阵 **V** 相乘，得到输出 **Z**，则单词 1 的输出向量 Z1 是只包含单词 1 信息的。

![](https://img-blog.csdnimg.cn/img_convert/56a925261b81ca2f74894dcb1087aefb.png) （Mask 之后的输出）

**第五步：**通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵 Zi ，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出 Zi 然后计算得到第一个 Multi-Head Attention 的输出 **Z**，**Z** 与输入 **X** 维度一样。

###  **3.2 第二个 Multi-Head Attention**

![](https://img-blog.csdnimg.cn/bd98a82aa8d04b34b2c3109ace3cc24f.png)

其实这块与上文 **Encoder** 中 的 Multi-Head Attention 具体实现细节上完全相同，区别在于 **Encoder** 的多头注意力里的 Q、K、V 是初始化多个不同的![](https://latex.csdn.net/eq?W%5E%7BQ%7D)，![](https://latex.csdn.net/eq?W%5E%7BK%7D)，![](https://latex.csdn.net/eq?W%5E%7BV%7D)矩阵得到的。而 **Decoder** 的 K、V 是来自于 **Encoder** 的输出，Q 是上层 Masked Self-Attention 的输出。

**Encoder** 中 的 Multi-Head Attention 只有一个输入，把此输入经过三个 linear 映射成 Q 、K 、V ， 而这里的输入有两个：

*   一个是 **Decoder** 的输入经过第一个大模块传过来的值
*   一个是 **Encoder** 最终结果

是把第一个值通过一个 linear 映射成了 Q，然后通过两个 linear 把第二个值映射成 K、V ，其它的与上文的完全一致。这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 **Mask**)。

###  3.3Linear 和 softmax

**Decoder** 最后会输出一个实数向量。那我们如何把浮点数变成一个单词？这便是线性变换层 **Linear 层**要做的工作，它之后就是 **Softmax 层**。

**Linear 层**是一个简单的全连接神经网络，它可以把 **Decoder** 产生的向量投射到一个比它大得多的、被称作对数几率（logits）的向量里。

![](https://img-blog.csdnimg.cn/20201208104251178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25vY21s,size_16,color_FFFFFF,t_70#pic_center)

不妨假设我们的模型从训练集中学习一万个不同的英语单词（我们模型的 “输出词表”）。因此对数几率向量为一万个单元格长度的向量——每个单元格对应某一个单词的分数。

接下来的 **Softmax 层**便会把那些分数变成概率（都为正数、上限 1.0）。概率最高的单元格被选中，并且它对应的单词被作为这个时间的输出。

![](https://img-blog.csdnimg.cn/7027bf7581be40d08ba074e7a733838d.png)

这张图片从底部以解码器组件产生的输出向量开始。之后它会转化出一个输出单词。

以上就是 Transformer 模型结构的全部解读了~

在这里如果想更清楚的了解，推荐大家看看大佬的讲解（感谢各位大佬，阿里嘎多！）

**b 站：**[【Transformer 从零详细解读 (可能是你见过最通俗易懂的讲解)】](?p=7&vd_source=725f2b2a52500df1eaed63206ebe0ab2)

**知乎：**[Transformer 模型详解（图解最完整版） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/338817680 "Transformer模型详解（图解最完整版） - 知乎 (zhihu.com)")

![](https://img-blog.csdnimg.cn/f621788939fd472980f5a377ab422226.gif)