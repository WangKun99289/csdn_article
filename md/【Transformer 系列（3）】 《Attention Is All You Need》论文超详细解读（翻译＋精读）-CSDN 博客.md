> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/130208816)

![](https://img-blog.csdnimg.cn/764f626e5f744604b5a4eea8d9e73538.jpeg)

前言
--

哒哒~ 时隔好久终于继续出论文带读了，这次回归当然要出一手王炸呀——没错，今天我们要一起学习的就是传说中的 **T****ransformer**！在 2021 年 [Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020) 一经论文《Attention is All You Need》提出，就如龙卷风一般震惊学术界，不仅在 NLP 领域大杀四方，在 CV 领域也是非常火，那我们就一起来看看这到底是何方神圣吧！

其实这篇论文我上周就读完了，但当时读的云里雾里，太多专业性语言看不懂，所以在这篇论文带读之前出了两篇分别介绍 encoder 和 [decoder](https://so.csdn.net/so/search?q=decoder&spm=1001.2101.3001.7020)（[【Transformer 系列（1）】encoder（编码器）和 decoder（解码器）](https://blog.csdn.net/weixin_43334693/article/details/130161845?spm=1001.2014.3001.5502 "【Transformer系列（1）】encoder（编码器）和decoder（解码器）")）以及注意力机制介绍（[【Transformer 系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解](https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5502 "【Transformer系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解")）建议小白看完这两篇了解了基础知识再来看这篇论文噢！

![](https://img-blog.csdnimg.cn/24c62513b96545269dd6889661cb8ca4.gif)

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀**前期回顾**

 [【Transformer 系列（1）】encoder（编码器）和 decoder（解码器）](https://blog.csdn.net/weixin_43334693/article/details/130161845?spm=1001.2014.3001.5502 "【Transformer系列（1）】encoder（编码器）和decoder（解码器）")

 [【Transformer 系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解](https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5502 " 【Transformer系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解")

 [【Transformer 系列（3）】 《Attention Is All You Need》论文超详细解读（翻译＋精读）](https://blog.csdn.net/weixin_43334693/article/details/130208816?spm=1001.2014.3001.5502 "【Transformer系列（3）】 《Attention Is All You Need》论文超详细解读（翻译＋精读）")

[【Transformer 系列（4）】Transformer 模型结构超详细解读](https://blog.csdn.net/weixin_43334693/article/details/130250571?spm=1001.2014.3001.5501 "【Transformer系列（4）】Transformer模型结构超详细解读")

**目录**

[前言](#t1)

[Abstract—摘要](#t2)

[一、Introduction—引言](#t3)

[二、Background—背景](#t4) 

[三、Model Architecture—模型结构](#t5)

[3.1 Encoder and Decoder Stacks—编码器栈和解码器栈](#t6)

[3.2 Attention—注意力机制](#t7)

[3.2.1 Scaled DotProduct Attention—缩放的点积注意力机制](#t8)

[3.2.2 MultiHead Attention—多头注意力机制](#t9)

[3.2.3 Applications of Attention in our Model—注意力机制在我们模型中的应用](#t10)

[3.3 Position-wise Feed-Forward Networks—基于位置的前馈神经网络](#t11)

[3.4 Embeddings and Softmax —词嵌入和 softmax](#t12)

[3.5 Positional Encoding——位置编码](#t13)

[四、Why Self-Attention—为什么选择 selt-attention](#t14)

[五、Training—训练](#t15)

[5.1 Training Data and Batching—训练数据和 batch](#t16)

[5.2 Hardware and Schedule—硬件和时间](#t17)

[5.3 Optimizer—优化器](#t18)

[5.4 Regularization—正则化](#t19)

[六、Results—结果](#t20)

[6.1 Machine Translation—机器翻译](#t21)

[6.2 Model Variations—模型变体](#t22)

[6.3 English Constituency Parsing—英文选区分析](#t23)

[七、Conclusion—结论](#t24)

![](https://img-blog.csdnimg.cn/870ff89a9bd0497d9842ecf8ecf641c1.gif)

**Abstract—摘要**
---------------

### 翻译

主流的序列转换模型都是基于复杂的循环[神经网络](https://so.csdn.net/so/search?q=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&spm=1001.2101.3001.7020 "神经网络")或卷积神经网络，且都包含一个 encoder 和一个 decoder。表现最好的模型还通过 attention 机制把 encoder 和 decoder 联接起来。我们提出了一个新的、简单的网络架构，Transformer. 它只基于单独的 attention 机制，完全避免使用循环和卷积。在两个翻译任务上表明，我们的模型在质量上更好，同时具有更高的并行性，且训练所需要的时间更少。我们的模型在 WMT2014 英语 - 德语的翻译任务上取得了 28.4 的 BLEU 评分。在现有的表现最好模型的基础上，包括整合模型，提高了 2 个 BLEU 评分。在 WMT2014 英语 - 德语的翻译任务上, 我们的模型在 8 个 GPU 上训练了 3.5 天（这个时间只是目前文献中记载的最好的模型训练成本的一小部分），创造了单模型的 SOTA 结果，BLEU 分数为 41.8，通过在大量和少量训练数据上所做的英语选区分析工作的成功，表明 Transformer 能很好的适应于其它任务。

### **精读**

**以前的方法**

主流序列转导模型基于复杂的 CNN 或 RNN，包括编码器和解码器。

有的模型使用注意力机制连接编码器和解码器，达到了最优性能。

**缺点：**

   ①难以并行

   ②时序中过早的信息容易被丢弃

   ③内存开销大

**本文方法**

本文提出的 Transformer 完全摒弃了之前的循环和卷积操作，完全基于注意力机制，拥有更强的并行能力，训练效率也得到较高提升。

**效果：**

在两个翻译任务上表明，我们的模型在质量上更好，同时具有更高的并行性，且训练所需要的时间更少。

**一、Introduction—引言**
---------------------

### **翻译**

RNN,LSTM,GRU,Gated Recurrent Neural Networks 在序列建模和转换任务上，比如语言模型和机器翻译，已经是大家公认的取得 SOTA 结果的方法。自此，无数的努力继续推动递归语言模型和 encoder-decoder 体系结构的界限。

  递归模型通常沿输入和输出序列的符号位置进行因子计算。在计算时将位置与步骤对齐，它们生成一系列隐藏状态 ht​，t 位置的 ht​使用它的前驱 ht−1​和当前的输入生成。这种内部的固有顺阻碍了训练样本的并行化，在序列较长时，这个问题变得更加严重，因为内存的限制限制了样本之间的批处理。最近的工作通过因子分解技巧 [21] 和条件计算 [32] 在计算效率方面取得了显著的提高，同时也提高了后者的模型性能。然而，顺序计算的基本约束仍然存在。

  在各种各样的任务中，注意力机制已经成为各种引人注目的序列模型和转换模型中的不可或缺的组成部分，它允许对依赖关系建模，而不需要考虑它们在输入或输出序列中的距离。然而，在除少数情况外的所有情况下 [27]，这种注意机制都与一个递归网络结合使用。

  在这项工作中，我们提出了 [Transformer](https://so.csdn.net/so/search?q=Transformer&spm=1001.2101.3001.7020 "Transformer")，这是一种避免使用循环的模型架构，完全依赖于注意机制来绘制输入和输出之间的全局依赖关系。Transformer 允许更显著的并行化，使用 8 个 P100 gpu 只训练了 12 小时，在翻译质量上就可以达到一个新的 SOTA。

### **精读**

**之前语言模型和机器翻译的方法和不足**

**方法：** RNN、LSTM、GRU、Encoder-Decoder

**不足：**

   ①从左到右一步步计算，因此难以并行计算

   ②过早的历史信息可能被丢弃，时序信息一步一步向后传递

   ③内存开销大，训练时间慢

**近期工作和问题**

近期一些工作通过分解技巧和条件计算提高了计算效率，但是顺序计算的本质问题依然存在

**本文改进**

**（1）引入注意力机制：** 注意力机制可以在 RNN 上使用，通过注意力机制把 encoder 的信息传给 decoder，可以允许不考虑输入输出序列的距离建模。

**（2）提出 Transformer ：** 本文的 Transformer 完全不用 RNN，这是一种避免使用循环的模型架构，**完全依赖于注意机制来绘制输入和输出之间的全局依赖关系**，并行度高，计算时间短。

**二、Background—背景** 
--------------------

### 翻译

减少序列计算的目标也成就了 Extended Neural GPU [16],ByteNet[18], 和 ConvS2S[9] 的基础, 它们都使用了卷积神经网络作为基础模块，并行计算所有输入和输出位置的隐藏表示。在这些模型中，将来自两个任意输入或输出位置的信号关联起来所需的操作数，随位置间的距离而增长，ConvS2S 为线性增长，ByteNet 为对数增长。这使得学习远距离位置之间的依赖性变得更加困难 [12]. 在 Transformer 中，这种情况被减少到了常数次操作，虽然代价是由于平均 注意力加权位置信息降低了有效分辨率，如第 3.2 节所述，我们用多头注意力抵消这种影响。

  self-attention, 有时也叫做内部注意力，是一种注意力机制，它将一个序列的不同位置联系起来，以计算序列的表示。self-attention 已经成功的运用到了很多任务上，包括阅读理解、抽象摘要、语篇蕴涵和学习任务无关的句子表征等。

  已经被证明，端到端的记忆网络使用循环 attention 机制替代序列对齐的循环，在简单的语言问答和语言建模任务中表现良好。

  然而，据我们所知，Transformer 是第一个完全依赖于 self-attetion 来计算其输入和输出表示而不使用序列对齐的 RNN 或卷积的转换模型，在下面的章节中，我们将描述 Transformer，motivate ，self-attention，并讨论它相对于 [17,18] 和[9]等模型的优势。

### **精读**

**CNN 代替 RNN**

**优点**：

    ①减小时序计算

    ②可以输出多通道

**问题：**

卷积的感受野是一定的，距离间隔较远的话就需要多次卷积才能将两个远距离的像素结合起来，所以对长时序来讲比较难。

**自注意力**

有时也称为内部注意力，是一种将单个序列的不同位置关联起来以计算序列表示的注意机制。 自注意力已成功用于各种任务，包括阅读理解、抽象摘要、文本蕴涵和学习与任务无关的句子表示。

**端到端记忆网络**

基于循环注意机制而不是序列对齐循环，并且已被证明在简单语言问答和语言建模任务中表现良好。

**Transformer 优点**

用注意力机制可以直接看一层的数据，就规避了 CNN 的那个缺点。

**三、Model Architecture—模型结构**
-----------------------------

### **翻译**

大多数有竞争力的序列转换模型都有 encoder-decoder 结构构。这里，encoder 将符号表示的输入序列 (x 1 , . . . , x n) 映射成一个连续表示的序列 z = ( z 1 , . . . , z n )。给定 z，解码器以一次生成一个字符的方式生成输出序列( y 1 , . . . , y m ) 。在每一步，模型都是自回归的[10]，在生成下一个字符时，将先前生成的符号作为附加输入。

  Transformer 遵循这个总体架构，使用堆叠的 self-attention 层、point-wise 和全连接层，分别用于 encoder 和 decoder，如图 1 的左半部分和右半部分所示。

![](https://img-blog.csdnimg.cn/6b35cf08fc5e492ca2703002d6db3049.png)

### 3.1 Encoder and Decoder Stacks—编码器栈和解码器栈

### 翻译

**Encoder**:encoder 由 N(N=6) 个完全相同的 layer 堆叠而成. 每层有两个子层。第一层是 multi-head self-attention 机制，第二层是一个简单的、位置全连接的前馈神经网络。我们在两个子层的每一层后采用残差连接 [11]，接着进行 layer normalization[1]。也就是说，每个子层的输出是 LayerNorm(x+Sublayer(x))，其中 Sublayer(x) 是由子层本身实现的函数。为了方便这些残差连接，模型中的所有子层以及 embedding 层产生的输出维度都为 dmodel​=512。

**Decoder**: decoder 也由 N(N=6) 个完全相同的 layer 堆叠而成. 除了每个编码器层中的两个子层之外，解码器还插入第三个子层，该子层对编码器堆栈的输出执行 multi-head attention 操作，与 encoder 相似，我们在每个子层的后面使用了残差连接，之后采用了 layer normalization。我们也修改了 decoder stack 中的 self-attention 子层，以防止当前位置信息中被添加进后续的位置信息。这种掩码与偏移一个位置的输出 embedding 相结合， 确保对第 i ii 个位置的预测 只能依赖小于 i 的已知输出。

### 精读

**编码器 encoder**

将一个长为 n 的输入（如句子），序列 (x1, x2, … xn) 映射为(z1, z2, …, zn)（机器学习可以理解的向量）

encoder 由 **n** 个相同层组成，**重复 6 个 layers**，每个 layers 会有两个 sub-layers，每个 sub-layers 里第一个 layer 是 multi-head attention，第二个 layer 是 simple，position-wise fully connected feed-forward network，简称 MLP。

每个 sub-layer 的输出都做一个残差连接和 layerNorm。计算公式：**LayerNorm(x + Sublayer(x) )**，Sublayer(x) 指 self-attention 或者 MLP。

残差连接需要输入和输出的维度一致，所以每一层的输出维度在 transformer 里都是固定的，都是 512 维。

与 CNN 不一样的是，**MLP 的空间维度是逐层下降，CNN 是空间维度下降，channel 维度上升。**

![](https://img-blog.csdnimg.cn/4c2710004d8540a9b1d166cd6df38d66.png)

**解码器 decoder**

decoder 拿到 encoder 的输出，会生成一个长为 m 的序列 (y1, y2, … , ym)。n 和 m 可以一样长、也可以不一样长，**编码时可以一次性生成，解码时只能一个个生成**（auto-regressive 自回归模型）

decoder 同样由 **n** 个相同层组成。

除了 encoder 中的两个子层外，decoder 还增加了一个子层：对 encoder 层的输出执行多头注意力。

另外对自注意力子层进行修改 (Mask)，防止某个 position 受后续的 position 的影响。确保位置 i 的预测只依赖于小于 i 的位置的已知输出。

输出就是标准的 **Linear+softmax**。

![](https://img-blog.csdnimg.cn/d8814a32e8954831a585cdb973d06f8e.png)

> 关于 encoder 和 decoder 更多详细了解请看：[【Transformer 系列（1）】encoder（编码器）和 decoder（解码器）](https://blog.csdn.net/weixin_43334693/article/details/130161845?spm=1001.2014.3001.5502 "【Transformer系列（1）】encoder（编码器）和decoder（解码器）") 

**LayerNorm 模块**

LayerNorm 是**层标准化**，和 BatchNorm 在很多时候几乎一样，除了实现方法不同。

BN 取的是不同样本的同一个特征，而 LN 取的是同一个样本的不同特征。在 BN 和 LN 都能使用的场景中，BN 的效果一般优于 LN，原因是基于不同数据，同一特征得到的归一化特征更不容易损失信息。

但是有些场景是不能使用 BN 的，例如 batchsize 较小或者在 RNN 中，这时候可以选择使用 LN，**LN 得到的模型更稳定且起到正则化的作用**。RNN 能应用到小批量和 RNN 中是因为 LN 的归一化统计量的计算是和 batchsize 没有关系的。

### 3.2 Attention—注意力机制

> 注意力机制详细了解请看这篇：[【Transformer 系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解](https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5501 "【Transformer系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解")

### 翻译

Attention 机制可以描述为将一个 query 和一组 key-value 对映射到一个输出，其中 query，keys，values 和输出均是向量。输出是 values 的加权求和，其中每个 value 的权重 通过 query 与相应 key 的兼容函数来计算。

![](https://img-blog.csdnimg.cn/fce514fdaef2419495fd69483eef8e53.png)

### **精读**

**概念**

注意力机制是对每个 Q 和 K 做内积，将它作为相似度。

当两个向量做内积时，如果他俩的 d 相同，向量内积越大，余弦值越大，相似度越高。

如果内积值为 0，他们是正交的，相似度也为 0。

**相关参数**

 **Q：** query(查询)

 **K：** key(键)

 **V：** value(值)

Q 就在目标 target 区域，就是 decoder 那块，K 和 V 都在源头 ，就是 encoder 区域。

自注意力则是 QKV 都在一个区域，要么都在 decoder 要么都在 encoder。 目的就是为了能够发现一句话内部或者一个时序内部的关联信息。

### 3.2.1 Scaled DotProduct Attention—缩放的点积注意力机制

### 翻译

我们称我们的特殊 attention 为 Scaled Dot-Product Attention(Figure 2)。输入由 query、![](https://latex.csdn.net/eq?d_%7Bk%7D)的 key 和![](https://latex.csdn.net/eq?d_%7Bv%7D)的 value 组成。我们计算 query 和所有 key 的点积，再除以![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bk%7D%7D)然后再通过 softmax 函数来获取 values 的权重。

  在实际应用中，我们把一组 query 转换成一个矩阵 Q，同时应用 attention 函数。key 和 valuue 也同样被转换成矩阵 K 和矩阵 V。我们按照如下方式计算输出矩阵：

![](https://img-blog.csdnimg.cn/cd37dcc5436841bfb7370b27e75a2fa1.png)

  additive attention 和 dot-product(multi-plicative) attention 是最常用的两个 attention 函数。dot-product attention 除了没有使用缩放因子![](https://latex.csdn.net/eq?%5Cfrac%7B1%7D%7B%5Csqrt%7Bd_%7Bk%7D%7D%7D)外，与我们的算法相同。Additive attention 使用一个具有单隐层的前馈神经网络来计算兼容性函数。尽管在理论上两者的复杂度相似，但是在实践中 dot-product attention 要快得多，而且空间效率更高，这是因为它可以使用高度优化的矩阵乘法代码来实现。

  当![](https://latex.csdn.net/eq?d_%7Bk%7D)的值较小时，这两种方法性能表现的相近，当![](https://latex.csdn.net/eq?d_%7Bk%7D)比较大时，addtitive attention 表现优于 dot-product attention。我们认为对于大的![](https://latex.csdn.net/eq?d_%7Bk%7D)​，点积在数量级上增长的幅度大，将 softmax 函数推向具有极小梯度的区域。为了抵消这种影响，我们对点积扩展![](https://latex.csdn.net/eq?%5Cfrac%7B1%7D%7B%5Csqrt%7Bd_%7Bk%7D%7D%7D)倍。

### **精读** 

**Scaled DotProduct Attention 简介**

Scaled Dot-Product Attention 是特殊 attention，输入包括查询 Q 和键 K 的维度 dk 以及值 V 的维度 dv​ 。计算查询和键的点积，将每个结果除![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bk%7D%7D) ，然后用 softmax() 函数来获得值的权重。 

在实际使用中，我们同时计算一组查询的注意力函数，并一起打包成矩阵 Q。键和值也一起打包成矩阵 K 和 V。 

![](https://img-blog.csdnimg.cn/f3891949bca24be4be7202e616aebe5e.png)

**加性 attention 和点积 (乘性)attention 区别**

   ①点积 attention 与我们的算法一致，除了缩放因子![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bk%7D%7D)

   ②加性 attention 使用带一个单隐层的前馈网络计算兼容函数

尽管这两种 attention 在原理复杂度上相似，但点积 attention 在实践中更快、空间效率更高，因为它可以使用高度优化的矩阵乘法代码。

> 为什么用 softmax？
> 
> 对于一个 Q 会给 n 个 K-V 对，Q 会和每个 K-V 对做内积，产生 n 个相似度。传入 softmax 后会得到 n 个非负并且和为 1 的权重值，把权重值与 V 矩阵相乘后得到注意力的输出。

> 为什么除以![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bk%7D%7D) ？
> 
> 虽然对于较小的 dk 两者的表现相似，但在较大的 dk 时，加法注意力要优于没有缩放机制的点乘注意力。我们认为在较大的 dk 时，点乘以数量级增长，将 softmax 函数推入梯度极小的区域，值就会更加向两端靠拢，算梯度的时候，梯度比较小。为了抵抗这种影响，我们使用 ![](https://latex.csdn.net/eq?%5Cfrac%7B1%7D%7B%5Csqrt%7Bd_%7Bk%7D%7D%7D)缩放点乘结果。

### 3.2.2 MultiHead Attention—多头注意力机制

### 翻译

相比于使![](https://latex.csdn.net/eq?d_%7Bmodel%7D)​维度的 queries,keys,values 执行一个 attention 函数，我们发现使用不同的学习到的线性映射把 queries, keys 和 values 线性映射到![](https://latex.csdn.net/eq?d_%7Bk%7D)，![](https://latex.csdn.net/eq?d_%7Bk%7D)和![](https://latex.csdn.net/eq?d_%7Bv%7D)​维度 h 次是有益的。在 queries,keys 和 values 的每个映射版本上，我们并行的执行 attention 函数，生成![](https://latex.csdn.net/eq?d_%7Bv%7D)维输出值。它们被拼接起来再次映射，生成一个最终值，如 Figure 2 中所示。

  Multi-head attention 允许模型把不同位置子序列的表示都整合到一个信息中。如果只有一个 attention head，它的平均值会削弱这个信息。

![](https://img-blog.csdnimg.cn/e2fe1d4b23324e26a14048c46d465b95.png)

 在这项工作中，我们采用 h = 8 个并行 attention 层或 head。 对每个 head，我们使用 ![](https://latex.csdn.net/eq?d_%7Bk%7D) = ![](https://latex.csdn.net/eq?d_%7Bv%7D) = ![](https://latex.csdn.net/eq?d_%7Bmodel%7D) / h = 64 ![](https://latex.csdn.net/eq?d_%7Bk%7D)= ![](https://latex.csdn.net/eq?d_%7Bv%7D) = ![](https://latex.csdn.net/eq?d_%7Bmodel%7D)/h = 64。 由于每个 head 尺寸上的减小，总的计算成本与具有全部维度的单个 head attention 相似。

### 精读

**方法**

不再使用一个 attention 函数，而是使用不同的学习到的线性映射将 queries，keys 和 values 分别线性投影到 dq、dk 和 dv 维度 h 次。

然后在 queries，keys 和 values 的这些投影版本中的每一个上并行执行注意力功能，产生 h 个注意力函数。

最后将这些注意力函数拼接并再次投影，产生最终输出值。

**目的**

一个 dot product 的注意力里面，没有什么可以学的参数。具体函数就是内积，为了识别不一样的模式，希望有不一样的计算相似度的办法。

**本文的点积注意力先进行了投影，而投影的权重 w 是可学习的**。多头注意力给 h 次机会学习不一样的投影方法，使得在投影进去的度量空间里面能够去匹配不同模式需要的一些相似函数，然后把 h 个头拼接起来，最后再做一次投影。这种做法有一些像卷积网络 CNN 的多输出通道。

多头注意力的输入还是 Q 、K 、V，但是输出是将不同的注意力头的输出合并，在投影到 W0 里，每个头 hi 把 Q,K,V 通过 可以学习的 Wq , Wk , Wv 投影到 dv 上，再通过注意力函数，得到 head。

### 3.2.3 Applications of Attention in our Model—注意力机制在我们模型中的应用

### 翻译

multi-head attention 在 Transformer 中有三种不同的使用方式：

*   在 encoder-decoder attention 层中，queries 来自前面的 decoder 层，而 keys 和 values 来自 encoder 的输出。这使得 decoder 中的每个位置都能关注到输入序列中的所有位置。 这是模仿序列到序列模型中典型的编码器—解码器的 attention 机制，例如 [38, 2, 9]。
*   encoder 包含 self-attention 层。 在 self-attention 层中，所有的 key、value 和 query 来自同一个地方，在这里是 encoder 中前一层的输出。 encoder 中的每个位置都可以关注到 encoder 上一层的所有位置。
*   类似地，decoder 中的 self-attention 层允许 decoder 中的每个位置都关注 decoder 层中当前位置之前的所有位置（包括当前位置）。 为了保持解码器的自回归特性，需要防止解码器中的信息向左流动。我们在 scaled dot-product attention 的内部 ，通过屏蔽 softmax 输入中所有的非法连接值（设置为 −∞）实现了这一点。

### 精读

Transformer 用了三种不同的注意力头：

**（1）Encoder 的注意力层：** 输入数据在经过 Embedding + 位置 encoding 后，复制成了三份一样的东西，分别表示 K Q V。同时这个数据既做 key 也做 query 也做 value，其实就是一个东西，所以叫自注意力机制。输入了 n 个 Q，每个 Q 会有一个输出，那么总共也有 n 个输出，输出是 V 加权和（权重是 Q 与 K 的相似度）。

![](https://img-blog.csdnimg.cn/5bb98c70cc2344209e1198ba32a22a7a.png)

**（2）Decoder 的注意力层：** 这个注意力层就不是自注意力了，其中 K 和 V 来自 Encoder 的输出，Q 来自掩码多头注意力输出。

![](https://img-blog.csdnimg.cn/f7856ed8f8654d6baa774b5e22770fe0.png)

**（3）Decoder 的掩码注意力层：** 掩码注意力层就是将 t 时刻后的数据权重设置为 0，该层还是自注意力的。

![](https://img-blog.csdnimg.cn/336e08247fb44765a72ed317192fbb6c.png)

###  3.3 Position-wise Feed-Forward Networks—基于位置的前馈神经网络

### 翻译

除了 encoder 子层之外，我们的 encder 和 decoder 中的每个层还包含一个全连接的前馈网络，该网络分别单独应用于每一个位置。这包括两个线性转换，中间有一个 ReLU 激活。

  尽管线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。 它的另一种描述方式是两个内核大小为 1 的卷积。 输入和输出的维度为 dmodel​ = 512，内部层的维度 dff​ = 2048。

### 精读

除了 attention 子层，我们 encoder-decoder 框架中**每一层都包含一个全连接的前馈网络**，它分别相同地应用于每个位置。它由两个线性变换和中间的一个 ReLU 激活函数组成

**公式**

![](https://img-blog.csdnimg.cn/f560942922e044a4b60891c1029d04d7.png)

两个线性转换作用于相同的位置，但是在他们用的参数不同。 另一种描述方式是比作两个 knernel size=1 的卷积核。输入输出的维度为 512，中间维度为 2048。

### 3.4 Embeddings and Softmax —词嵌入和 softmax

### 翻译

与其他序列转换模型类似，我们使用学习到的嵌入词向量 将输入字符和输出字符转换为维度为 dmodel​的向量。我们还使用普通的线性变换和 softmax 函数将 decoder 输出转换为预测的下一个词符的概率。在我们的模型中，两个嵌入层之间和 pre-softmax 线性变换共享相同的权重矩阵，类似于 [30]。 在嵌入层中，我们将这些权重乘以![](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bmodel%7D%7D)

### 精读

> **Embedding：** 特征嵌入，embedding 是可以简单理解为通过某种方式将词向量化，即输入一个词输出该词对应的一个向量。（embedding 可以采用训练好的模型如 GLOVE 等进行处理，也可以直接利用深度学习模型直接学习一个 embedding 层，Transformer 模型的 embedding 方式是第二种，即自己去学习的一个 embedding 层。）

**本文方法**

embeddings 将输入和输出 tokens 转换为向量，线性变换和 softmax 函数将 decoder 输出转换为预测的写一个 token 概率。

### 3.5 Positional Encoding——位置编码

### 翻译

由于我们的模型不包含循环或卷积，为了让模型利用序列的顺序信息，我们必须加入序列中关于字符相对或者绝对位置的一些信息。 为此，我们在 encoder 和 decoder 堆栈底部的输入嵌入中添加 “位置编码”。 位置编码和嵌入的维度 dmodel​相同，所以它们两个可以相加。有多种位置编码可以选择，例如通过学习得到的位置编码和固定的位置编码 [9]

在这项工作中，我们使用不同频率的正弦和余弦函数:

![](https://img-blog.csdnimg.cn/0824a9e8d0fd4b18a08d540c86282de1.png)

其中 pos 是位置，i 是维度。也就是说，位置编码的每个维度对应于一个正弦曲线。波长形成了从 2π到 10000·2π的几何数列。我们之所以选择这个函数，是因为我们假设它可以让模型很容易地通过相对位置来学习, 因为对任意确定的偏移 k, PEpos+k​可以表示为 PEpos​的线性函数。

  我们还尝试使用预先学习的 positional embeddings[9] 来代替正弦波，发现这两个版本产生了几乎相同的结果 (see Table 3 row (E))。我们之所以选择正弦曲线，是因为它允许模型扩展到比训练中遇到的序列长度更长的序列。

### 精读

**目的**

因为 transformer 模型不包含循环或卷积，输出是 V 的加权和（权重是 Q 与 K 的相似度，与序列信息无关），对于任意的 K-V，将其打乱后，经过注意力机制的结果都一样

但是它顺序变化而值不变，在处理时序数据的时候，**一个序列如果完全被打乱，那么语义肯定发生改变，而注意力机制却不会处理这种情况。**

**方法**

在注意力机制的**输入中加入时序信息**，位置在 encoder 端和 decoder 端的 embedding 之后，用于补充 Attention 机制本身不能捕捉位置信息的缺陷。

一次词在嵌入层表示成一个 512 维的向量，用另一个 512 维的向量表示位置数字信息的值。用周期不一样的 sin 和 cos 函数计算。

四、Why Self-Attention—为什么选择 selt-attention
-----------------------------------------

### 翻译

在这一节中，我们将 self-attention layers 与常用的 recurrent layers 和 convolutional layers 进行各方面的比较，比较的方式是 将一个可变长度的符号表示序列 (x 1 , . . . , x n) 映射到另一个等长序列 ( z 1 , . . . , z n ) ，用 xi​,zi​∈Rd，比如在典型的序列转换的 encoder 或 decoder 中的隐藏层。我们考虑三个方面，最后促使我们使用 self-attention。

  一是每层的总计算复杂度。另一个是可以并行化的计算量，以所需的最小序列操作数衡量。

  第三个是网络中长距离依赖关系之间的路径长度。在许多序列转换任务中，学习长距离依赖性是一个关键的挑战。影响学习这种依赖关系能力的一个关键因素是网络中向前和向后信号必须经过的路径的长度。输入和输出序列中任意位置组合之间的这些路径越短，越容易学习长距离依赖。因此，我们还比较了在由 different layer types 组成的网络 中的任意两个输入和输出位置之间的最大的路径长度。

  如表 1 所示，self-attention layer 用常数次 (O ( 1) ) 的操作连接所有位置，而 recurrent layer 需要 O(n)顺序操作。在计算复杂度方面，当序列长度 N 小于表示维度 D 时，self-attention layers 比 recurrent layers 更快，这是使用最先进的机器翻译模型表示句子时的常见情况，例如 word-piece [38] 和 byte-pair [31] 表示。为了提高包含很长序列的任务的计算性能，可以仅在以输出位置为中心，半径为 r 的的领域内使用 self-attention。这将使最大路径长度增长到 O ( n / r ) 。我们计划在今后的工作中进一步研究这种方法。

  核宽度为 k<n 的单层卷积不会连接每一对输入和输出的位置。要这么做，在相邻的内核情况下，需要一个 n 个卷积层的堆栈， 在扩展卷积的情况下需要 O(logk(n)) 层 [18]，它们增加了网络中任意两个位置之间的最长路径的长度。 卷积层通常比循环层代价更昂贵，这与因子 k 有关。然而，可分卷积[6] 大幅减少复杂度到 O(k⋅n⋅d+n⋅d2)。然而，即使 k=n，可分离卷积的复杂度等于 self-attention layer 和 point-wise feed-forward layer 的组合，这是我们在模型中采用的方法。

  一个随之而来的好处是，self-attention 可以产生更多可解释的模型。我们从我们的模型中研究 attention 的分布，并在附录中展示和讨论示例。每个 attention head 不仅清楚地学习到执行不同的任务，还表现出了许多和句子的句法和语义结构相关的行为。

### 精读

**考虑因素**

一是**每层的总计算复杂度**

二是可以并行化的计算量，以**所需的最小序列操作数**衡量

三是网络中**长距离依赖关系之间的路径长度**

**研究关键**

计算远距离依赖一直是序列转换任务中的关键挑战，其中的一个关键因素就是其路径长度。

路径距离越短，学习长距离依赖越容易。

**和 CNN、RNN 比较**

n 表示序列长度，d 是隐藏层维度，k 表示卷积核尺寸，r 表示受限自注意力的窗口大小

![](https://img-blog.csdnimg.cn/5e9e7e6b735a4dc5a24bf7588dca5ab5.png) ![](https://img-blog.csdnimg.cn/3d9eb2a5f8a6469589062d30d47f16a0.png)![](https://img-blog.csdnimg.cn/b4ce72cdc50a4e35acf27bfa25f03aa1.png)

**结论**

self-attention 能够产生解释性更强的模型

**五、Training—训练**
-----------------

### 5.1 Training Data and Batching—训练数据和 batch

### **翻译**

我们在标准的 WMT 2014 英语 - 德语数据集上进行了训练，其中包含约 450 万个句子对。 这些句子使用 byte-pair 编码 [3] 进行编码，源语句和目标语句共享大约 37000 个词符的词汇表。 对于英语 - 法语翻译，我们使用大得多的 WMT 2014 英法数据集，它包含 3600 万个句子，并将词符分成 32000 个 word-piece 词汇表[38]。 序列长度相近的句子一起进行批处理。 每个训练批次的句子对包含大约 25000 个源词符和 25000 个目标词符。

### 精读

**英语 - 德语**

 **数据集：** 标准的 WMT 2014 英语 - 德语数据集

 **句子对数量：** 约 450 万个句子对

 **训练方式：** byte-pair 编码

**英语 - 法语**

 **数据集：** WMT 2014 英法数据集

 **句子对数量：** 3600 万个句子对

 **训练方式：** 序列长度相近的句子一起进行批处理

 **batch：** 每个训练批次的句子对包含大约 25000 个源词符和 25000 个目标词符

### 5.2 Hardware and Schedule—硬件和时间

### 翻译

我们在一台具有 8 个 NVIDIA P100 gpu 的机器上训练我们的模型。对于 paper 中描述的使用超参数的基础模型，每个训练步骤大约需要 0.4 秒。我们对基础模型进行了总共 100000 步或 12 小时的训练。对于我们的大型模型（见表 3 的底线），步进时间为 1.0 秒。大模型 使用了 30 万步（3.5 天）的训练。

### 精读

**硬件**

一台带有 8 个 NVIDIA P100 GPUs 的机器

**时间**

**基础模型：** 每个训练步骤耗时 0.4s。需要 100000 步（12h）。

**大模型：** 每步是 1.0s，需要 300000 步（3.5 天）。

### 5.3 Optimizer—优化器

### 翻译

我们使用 Adam 优化器 [20]，其中β1 = 0.9, β2 = 0.98 及ϵ= 10-9。 

  这对应于在第一次 warmup_steps 步骤中线性地增加学习速率，并且随后将其与步骤数的平方根成比例地减小。 我们使用 warmup_steps=4000。

### 精读

**优化器：** Adam 优化器

**参数：** β1 = 0.9, β2 = 0.98 及ϵ= 10-9

**公式：**

**![](https://img-blog.csdnimg.cn/543147f648244dc9b1e1f2aa42519bd2.png)**

**warmup_steps：** 4000

### 5.4 Regularization—正则化

### 翻译

训练中我们采用三种正则化：

**Residual Dropout** 我们在对每个子层的输出上执行 dropout 操作, 这个操作在 additive 操作（子层的输出加上子层的输入）和 normalized 操作之前。 此外，在编码器和解码器堆栈中，我们将丢弃应用到嵌入和位置编码的和。 对于基础模型，我们使用 Pdrop​ = 0.1 丢弃率。

![](https://img-blog.csdnimg.cn/20191113145936410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25vY21s,size_16,color_FFFFFF,t_70#pic_center)

**Label Smoothing** 在训练过程中，我们采用了值εls​=0.1[36] 的标签平滑。这会影响 ppl, 因为模型学习到了更多的不确定性，但提高了准确率和 BLEU 评分。

### 精读

**Residual Dropout**

 在 additive 操作和 normalized 操作之前执行 dropout 操作，但凡是带有权重的层，都有 dropout=0.1

**Label Smoothing**

使用交叉熵计算损失时，只考虑到训练样本中正确的标签位置的损失，而忽略了错误标签位置的损失。Label Smoothing 可以有效避免上述错误，训练中使用了 0.1 的标签平滑，使得模型学到了更多的不确定性。

**六、Results—结果**
----------------

### 6.1 Machine Translation—机器翻译

### 翻译

在 WMT 2014 英语 - 德语翻译任务中，大型 Transformer 模型（表 2 中的 Transformer (big)）比以前报道的最佳模型（包括整合模型）高出 2 个以上的 BLEU 评分，以 28.4 分建立了一个全新的 SOTA BLEU 分数。 该模型的配置列在表 3 的底部。 在 8 个 P100 GPU 上花费 3.5 天进行训练。 即使我们的基础模型也超过了以前发布的所有模型和整合模型，且训练成本只是这些模型的一小部分。

  我们的模型在 WMT2014 英语 - 德语的翻译任务上取得了 28.4 的 BLEU 评分。在现有的表现最好模型的基础上，包括整合模型，提高了 2 个 BLEU 评分。  
    
  在 WMT 2014 英语 - 法语翻译任务中，我们的大型模型的 BLEU 得分为 41.0，超过了之前发布的所有单一模型，训练成本低于先前最先进模型的 1 ∕ 4 。 英语 - 法语的 Transformer (big) 模型使用 Pdrop​=0.1，而不是 0.3。

  对于基础模型，我们使用的单个模型来自最后 5 个 checkpoints 的平均值，这些 checkpoints 每 10 分钟保存一次。 对于大型模型，我们对最后 20 个 checkpoints 进行了平均。 我们使用 beam search，beam 大小为 4 ，长度惩罚α = 0.6 [38]。 这些超参数是在开发集上进行实验后选定的。 在推断时，我们设置最大输出长度为输入长度 + 50，但在条件允许时会尽早终止 [38]。

  表 2 总结了我们的结果，并将我们的翻译质量和训练成本与文献中的其他模型体系结构进行了比较。 我们通过将训练时间、所使用的 GPU 的数量以及每个 GPU 的持续单精度浮点能力的估计相乘来估计用于训练模型的浮点运算的数量。

### 精读

**WMT 2014 英语 - 德语翻译任务表现**

**（1）评分更高：** 取得了 28.4 的 BLEU 评分。在现有的表现最好模型的基础上，包括整合模型，提高了 2 个 BLEU 评分。

**（2）成本更小：** 训练成本只是这些模型的一小部分

**WMT 2014 英语 - 法语翻译任务表现**

**（1）评分更高：** 大型模型的 BLEU 得分为 41.0，超过了之前发布的所有单一模型

**（2）成本更小：** 训练成本低于先前最先进模型的 1 ∕ 4

### **6.2 Model Variations—模型变体**

### **翻译**

为了评估 Transformer 不同组件的重要性，我们以不同的方式改变我们的基础模型，观测在开发集 newstest2013 上英文 - 德文翻译的性能变化。 我们使用前一节所述的 beam search，但没有平均 checkpoint。 我们在表中列出这些结果 3.

  在表 3 的行（A）中，我们改变 attention head 的数量和 attention key 和 value 的维度，保持计算量不变，如 3.2.2 节所述。 虽然只有一个 head attention 比最佳设置差 0.9 BLEU，但质量也随着 head 太多而下降。

  在表 3 行（B）中，我们观察到减小 key 的大小 dk​会有损模型质量。 这表明确定兼容性并不容易，并且比点积更复杂的兼容性函数可能更有用。 我们在行（C）和（D）中进一步观察到，如预期的那样，更大的模型更好，并且 dropout 对避免过度拟合非常有帮助。 在行（E）中，我们用学习到的 positional encoding[9] 来替换我们的正弦位置编码，并观察到与基本模型几乎相同的结果。

![](https://img-blog.csdnimg.cn/566732e1017040ad9e65c769cfdd3fc8.png)

### **精读**

**目的**

为了评估 Transformer 不同组件的重要性，我们以不同的方式改变我们的基础模型，观测在开发集 newstest2013 上英文 - 德文翻译的性能变化。

**措施 1**

改变 attention head 的数量和 attention key 和 value 的维度，保持计算量不变

**结果：** 虽然只有一个 head attention 比最佳设置差 0.9 BLEU，但质量也随着 head 太多而下降。

**措施 2**

减小 key 的大小

**结果：** dk 会有损模型质量。 这表明确定兼容性并不容易，并且比点积更复杂的兼容性函数可能更有用。

**措施 3**

换更大的模型，并增加 dropout

**结果：** 更大的模型更好，并且 dropout 对避免过度拟合非常有帮助。

**措施 4**

用学习到的 positional encoding 来替换正弦位置编码

**结果：** 与基本模型几乎相同的结果

### 6.3 English Constituency Parsing—英文选区分析

### 翻译

为了评估 Transformer 是否可以扩展到其他任务，我们进行了英语选区解析的实验。这项任务提出特别的挑战：输出受到很强的结构性约束，并且比输入要长很多。 此外，RNN 序列到序列模型还没有能够在小数据 [37] 中获得最好的结果。

  我们用 dmodel​ = 1024 在 Penn Treebank[25] 的 Wall Street Journal（WSJ）部分训练了一个 4 层的 transformer，约 40K 个训练句子。 我们还使用更大的高置信度和 BerkleyParser 语料库，在半监督环境中对其进行了训练，大约 17M 个句子 [37]。 我们使用了一个 16K 词符的词汇表作为 WSJ 唯一设置，和一个 32K 词符的词汇表用于半监督设置。

  我们只在开发集的 Section 22 上进行了少量的实验来选择 dropout、attention 和 residual（第 5.4 节）、learning rates 和 beam size，所有其他参数从英语到德语的基础翻译模型保持不变。在推断过程中，我们将最大输出长度增加到输入长度 + 300。 对于 WSJ 和半监督设置，我们都使用 beam size = 21 和α = 0.3 。

  表 4 中我们的结果表明，尽管缺少特定任务的调优，我们的模型表现得非常好，得到的结果比之前报告的 Recurrent Neural Network Grammar [8] 之外的所有模型都好。

  与 RNN 序列到序列模型 [37] 相比，即使仅在 WSJ 训练 40K 句子组训练时，Transformer 也胜过 BerkeleyParser [29]。

### 精读

**做法**

（1）用 d model ​ = 1024 在 Penn Treebank 的 Wall Street Journal（WSJ）部分训练了一个 4 层的 transformer，约 40K 个训练句子。

（2）使用更大的高置信度和 BerkleyParser 语料库，在半监督环境中对其进行了训练，大约 17M 个句子。

（3）使用了一个 16K 词符的词汇表作为 WSJ 唯一设置，和一个 32K 词符的词汇表用于半监督设置。

**结果**

尽管缺少特定任务的调优，我们的模型表现得非常好，得到的结果比之前报告的 Recurrent Neural Network Grammar 之外的所有模型都好。

七、Conclusion—结论
---------------

### 翻译

在这项工作中，我们提出了 Transformer，第一个完全基于 attention 的序列转换模型，用 multi-headed self-attention 取代了 encoder-decoder 架构中最常用的 recurrent layers。

  对于翻译任务，Transformer 比基于循环或卷积层的体系结构训练更快。 在 WMT 2014 英语 - 德语和 WMT 2014 英语 - 法语翻译任务中，我们取得了最好的结果。 在前面的任务中，我们最好的模型甚至胜过以前报道过的所有整合模型。

  我们对基于 attention 的模型的未来感到兴奋，并计划将它们应用于其他任务。 我们计划将 Transformer 扩展到除文本之外的涉及输入和输出模式的问题，并研究局部的、受限的 attention 机制，以有效地处理图像、音频和视频等大型输入和输出。 让生成具有更少的顺序性是我们的另一个研究目标。

  我们用于训练和评估模型的代码可以在 [GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.](https://github.com/tensorflow/tensor2tensor "GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.") 找到。

### 精读

**再次介绍 transformer**

是第一个仅基于注意力机制的序列转录模型，使用多头自注意力机制替换了目前在编码器 - 译码器结构中最普遍使用的循环层。

对于机器翻译任务，Transformer 模型训练速度相较于其他模型显著提高，实现了新高。

**未来应用**

将使用在其他序列模型训练任务中，不限于图片、音频和视频。

这篇论文我们就一起读完了，本篇文章创新点在于抛弃了之前传统的 encoder-decoder 模型必须结合 cnn 或者 rnn 的固有模式，只用 Attention。文章的主要目的在于减少计算量和提高并行效率的同时不损害最终的实验结果。因为涉及太多 [NLP](https://so.csdn.net/so/search?q=NLP&spm=1001.2101.3001.7020) 领域，很多触及到我的知识盲点，所以读得不是很细，以后学习中会再逐渐完善细节滴~

![](https://img-blog.csdnimg.cn/f56e111e2a604dd2a92fc2b59dad2c5b.gif)