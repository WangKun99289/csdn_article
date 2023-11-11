> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_43334693/article/details/130313746)

![](https://img-blog.csdnimg.cn/0a64a64e43b14ac0a0c86d30d5b3e78a.jpeg)

前言 
---

前面几篇我们一起读了 [transformer](https://so.csdn.net/so/search?q=transformer&spm=1001.2101.3001.7020) 的论文，更进一步了解了它的模型架构，这一篇呢，我们就来看看它是如何代码实现的！

![](https://img-blog.csdnimg.cn/9566daf425834f23a0ea03637057ff4d.png)

（建议大家在读这一篇之前，先去看看上一篇[模型结构讲解](https://blog.csdn.net/weixin_43334693/article/details/130250571?spm=1001.2014.3001.5501 "模型结构讲解 ")  这样可以理解更深刻噢！）

transformer 代码有很多版本，本文是参考 B 站[这位大佬](https://www.bilibili.com/video/BV1dR4y1E7aL/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=31a0246bd070fa6719bbec1ba2154865 "这位大佬")改进后的代码进行解读，因为我也是刚开始学习，能力有限，如有不详实之处，大家可移步至文末的传送门去看大佬解读的更多细节嗷~

![](https://img-blog.csdnimg.cn/f5f7feb7e99b4a3295d4f6f71505cc94.gif)

![](https://img-blog.csdnimg.cn/962f7cb1b48f44e29d9beb1d499d0530.gif)​   🍀**前期回顾**

 [【Transformer 系列（1）】encoder（编码器）和 decoder（解码器）](https://blog.csdn.net/weixin_43334693/article/details/130161845?spm=1001.2014.3001.5502 "【Transformer系列（1）】encoder（编码器）和decoder（解码器）")

[【Transformer 系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解](https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5502 "【Transformer系列（2）】注意力机制、自注意力机制、多头注意力机制、通道注意力机制、空间注意力机制超详细讲解")  
[【Transformer 系列（3）】《Attention Is All You Need》论文超详细解读（翻译＋精读）](https://blog.csdn.net/weixin_43334693/article/details/130208816?spm=1001.2014.3001.5502 "【Transformer系列（3）】《Attention Is All You Need》论文超详细解读（翻译＋精读）")  
[【Transformer 系列（4）】Transformer 模型结构超详细解读](https://blog.csdn.net/weixin_43334693/article/details/130250571?spm=1001.2014.3001.5501 "【Transformer系列（4）】Transformer模型结构超详细解读")

**目录**

[前言](#t1) 

[🚀0. 导入依赖库](#t2)

[🚀1. 数据预处理](#t3) 

[1.1 数据准备](#t4)

[1.1.1 训练集：句子输入部分](#t5)

[1.1.2 测试集：构建词表](#t6)

[1.2 数据构建](#t7) 

[1.2.1 实现一个 minibatch 迭代器](#t8)

[1.2.2 自定义一个 MyDataSet 去读取这些句子](#t9)

[🚀2. 模型整体架构](#t10)

[2.1 超参数设置](#t11)

[2.2 整体架构](#t12)

[2.3 模型训练](#t13)

[🚀3. 编码器（Encoder）](#t14)

[3.1 Encoder Layer：单个编码器层](#t15)

[3.2 Encoder：编码器](#t16)

[3.3 Padding Mask：形成一个符号矩阵](#t17)

[🚀4. 解码器（Decoder）](#t18)

[4.1Decoder Layer：单个解码层](#t19) 

[4.2Decoder：解码器](#t20)

[4.3 Sequence Mask：屏蔽子序列的 mask](#t21)

[🚀5. 位置编码（Position Embedding）](#t22)

[🚀6. 注意力机制（Attention）](#t23)

[6.1Scaled DotProduct Attention：缩放点积注意力机制](#t24)

[6.2MultiHead Attention：多头注意力机制](#t25)

[🚀7. 前馈神经网络（PoswiseFeedForward）](#t26)

          [7.1 实现方式 1：Conv1d](#7.1%20%E5%AE%9E%E7%8E%B0%E6%96%B9%E5%BC%8F1%EF%BC%9AConv1d)

 [7.2 实现方式 2：Linear](#t27)

![](https://img-blog.csdnimg.cn/349c1282ff7546ecab74c7a0978db491.gif)

🚀0. 导入依赖库
----------

```
#======================0.导入依赖库=============================#
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
```

*   **numpy：**  科学计算库，提供了矩阵，线性代数，傅立叶变换等等的解决方案, 最常用的是它的 N 维数组对象
*   **torch：** 这是主要的 Pytorch 库。它提供了构建、训练和评估神经网络的工具
*   **torch.nn：**  torch 下包含用于搭建神经网络的 modules 和可用于继承的类的一个子包
*   **torch.optim：**  优化器 Optimizer。主要是在模型训练阶段对模型可学习参数进行更新，常用优化器有 SGD，RMSprop，Adam 等
*   **matplotlib.pyplot：**  matplotlib.pyplot 是一个命令型函数集合，它可以像使用 Matlab 一样使用 matplotlib，pyplot 中的每一个函数都会对画布图像作出相应的改变
*   **math：**  调用这个库进行数学运算

🚀1. 数据预处理 
-----------

本文以一个简单的**德语到英语**的**机器翻译**任务 Demo 为例。

### 1.1 数据准备

#### **1.1.1 训练集：句子输入部分**

```
# ===1.训练集（句子输入部分）===#
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
```

*   第一个德语句子 'ich mochte ein bier P'  --> **编码端的输入**
*   第二个英语句子 'S i want a beer' --> **解码端的输入**
*   第三个英语句子 'i want a beer E' --> **解码端的真实标签（答案）**

可以通过这个图来理解一下：

![](https://img-blog.csdnimg.cn/c49e1526a6b04eeca62a63728343dc33.png)

> P、S、E 是什么？
> 
> *   **P：**pad 字符。如果当前批次的数据量小于时间步数，将填写空白序列的符号。
> *   **S：**Start。显示 解码 输入开始 的符号 
> *   **E：**End。显示 解码 输出开始 的符号

#### **1.1.2 测试集：构建词表**

```
#===2.测试集（构建词表）===#
    
    # 编码端的词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)  # src_vocab_size：实际情况下，它的长度应该是所有德语单词的个数
 
    # 解码端的词表
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)  # 实际情况下，它应该是所有英语单词个数
```

词嵌入本身是一个 look up 查表的过程，因此需要构建词表：token 及其索引。

现在的实际任务中，一般使用 Huggingface Transformers 库的 Tokenizer 等 API 直接获取。

其实编码端和解码端可以共用一个词表的。

### **1.2 数据构建** 

#### **1.2.1 实现一个 minibatch 迭代器**

```
def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]] # 输入数据集
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]] # 输出数据集
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]] # 目标数据集
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)
```

这段代码主要是把字符串类型的文本转成词表索引，然后再把索引转成 tensor 类型。

#### **1.2.2** 自定义一个 MyDataSet 去读取这些句子

```
​ class MyDataSet(Data.Dataset):
     """自定义DataLoader"""
 ​
     def __init__(self, enc_inputs, dec_inputs, dec_outputs):
         super(MyDataSet, self).__init__()
         self.enc_inputs = enc_inputs
         self.dec_inputs = dec_inputs
         self.dec_outputs = dec_outputs
 ​
     def __len__(self):
         return self.enc_inputs.shape[0]
 ​
     def __getitem__(self, idx):
         return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
 ​
 ​
 loader = Data.DataLoader(
     MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
```

 我们需要在自定义的数据集类中继承 Dataset 类，同时还需要实现两个方法：

*   **__len__方法：**  能够实现通过全局的 len()  方法获取其中的元素个数
*   **__getitem__方法：**  能够通过传入索引的方式获取数据，例如通过 dataset[i]  获取其中的第 i 条数据

最后 DataLoader 进行封装：dataset，batch_size，shuffle(是否打乱)

🚀2. 模型整体架构
-----------

### 2.1 超参数设置

```
src_len = 5 # length of source 编码端的输入长度
    tgt_len = 5 # length of target 解码端的输入长度
 
    #===Transformer参数===#
    d_model = 512  # Embedding Size 每一个字符转化成Embedding的大小
    d_ff = 2048  # FeedForward dimension 前馈神经网络映射到多少维度
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer  encoder和decoder的个数，这个设置的是6个encoder和decoder堆叠在一起（encoder和decoder的个数必须保持一样吗）
    n_heads = 8  # number of heads in Multi-Head Attention  多头注意力机制时，把头分为几个，这里说的是分为8个
```

**重要参数：**

*   **src_len ：**  编码端的输入长度
*   **tgt_len ： **  解码端的输入长度
*   **d_model：** 需要定义 embeding 的维度，论文中设置的 512
*   **d_ff：** FeedForward 层隐藏神经元个数，论文中设置的 2048
*   **d_k = d_v：** Q、K、V 向量的维度，其中 Q 与 K 的维度必须相等，V 的维度没有限制，都设为 64
*   **n_layers：**  Encoder 和 Decoder 的个数，也就是图中的 Nx
*   **n_heads：** 多头注意力中 head 的数量

### 2.2 整体架构

```
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder=Encoder().to(device) # 编码层
        self.decoder=Decoder().to(device) # 解码层
       # 输出层，d_model是解码层每一个token输出维度的大小，之后会做一个tgt_vocab_size大小的softmax
        self.projection=nn.Linear(d_model,tgt_vocab_size,bias=False).to(device)
    
    # 实现函数
    def forward(self, enc_inputs,dec_inputs):
        """
         Transformers的输入：两个序列（编码端的输入，解码端的输入）
         enc_inputs: [batch_size, src_len]   形状：batch_size乘src_len
         dec_inputs: [batch_size, tgt_len]   形状：batch_size乘tgt_len
         """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
 
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs,enc_self_attns=self.encoder(enc_inputs)
 
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs,dec_self_attns,dec_enc_attns=self.decoder(dec_inputs,enc_inputs,enc_outputs)
 
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits=self.projection(dec_outputs)
        return dec_logits.view(-1,dec_logits.size(-1)),enc_self_attns,dec_self_attns,dec_enc_attns
```

Transformer 主要就是**调用 Encoder 和 Decoder**。最后返回 dec_logits 的维度是 **[batch_size * tgt_len, tgt_vocab_size]**。

**主要流程：**

*   输入文本进行词嵌入和位置编码，作为最终的文本嵌入；
*   文本嵌入经过 Encoder 编码，得到注意力加权后输出的编码向量以及自注意力权重矩阵；
*   然后将编码向量和样本的 Ground trurh 共同输入解码器，经过注意力加权等操作后输出最终的上下文向量，然后映射到词表大小的线性层上进行解码生成文本；
*   最终返回代表预测结果的 logits 矩阵。

 **主要参数：**  

*   **d_model：**解码层每个 token 输出的维度大小，之后会做一个 tgt_vocab_size 大小的 softmax

>  **d_model：**预测一个德语单词被翻译成英语，它会对应为那个单词，这里输入就是一个单词在词表中的维度，这里的维度是 512，所以在词表中一个单词的维度是 512。
> 
> 如果一句话有 n 个单词，那么在翻译的整个过程中就会调用 n 次这个全连接函数。
> 
> 举个栗子：英语单词有 100000 个，那么这儿的 tgt_vocab_size 就是 1000000 个。到达这儿，就好像是一个分类任务，看这个单词属于这 100000 个类中的哪一个类，最后全连接分类的结果然后再进行一个 softmax 就会得到这 100000 个单词每个单词的概率。哪个单词的概率最大，那么我们就把这个德语单词翻译成那个单词。也就是我们这儿的 projection 就是那个德语单词被翻译成英语单词的词。

*   **enc_inputs：**编码端输入。形状为 [batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部 tokens 的输出，可以是特定每一层的输出；也可以是中间某些参数的输出
*   **enc_outputs：**编码端输出。编码端的输入 (enc_inputs) 通过编码端（encoder）流到编码端的输出(enc_outputs)
*   **enc_self_attns：**Q、K 转置相乘后 softmax 的矩阵值，代表每个单词和其他单词的相关性
*   **dec_outputs：** 解码端输出，用于后续的 linear 映射
*   **dec_self_attns：** 类比于 enc_self_attns 是查看每个单词对解码端中输入的其余单词的相关性
*   **dec_enc_attns：**解码端中每个单词对 encoder 中每个单词的相关性
*   **dec_logits.view：** 进行 view 操作主要是为了适应后面的 CrossEntropyLoss（交叉熵损失函数） API 的参数要求

### 2.3 模型训练

```
model=Transformer().to(device) # 调用Transformer模型
criterion=nn.CrossEntropyLoss(ignore_index=0) # 交叉熵损失函数
optimizer=optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)# 用Adam的话效果不好
 
for epoch in range(epochs):
    for enc_inputs,dec_inputs,dec_outputs in loader:
        """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
        """
        enc_inputs,dec_inputs,dec_outputs=enc_inputs.to(device),dec_inputs.to(device),dec_outputs.to(device)
 
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs,enc_self_attns,dec_self_attns,dec_enc_attns=model(enc_inputs,dec_inputs)
 
        # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        loss=criterion(outputs,dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

这里的损失函数里面设置了一个参数**`ignore_index=0`**，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）

🚀3. [编码器](https://so.csdn.net/so/search?q=%E7%BC%96%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)（Encoder）
--------------------------------------------------------------------------------------------------------

编码器 (Encoder）由三个部分组成：**输入**、**多头注意力**、**前馈神经网络**。

![](https://img-blog.csdnimg.cn/1c8c3afebedb4fd485c8711415a0be77.png)

**流程**

*   输入文本的索引 tensor，经过词嵌入层得到词嵌入，然后和位置编码线性相加作为输入层的最终输出；
*   随后，每一层的输出最为下一层编码块的输入，在每个编码块里进行注意力计算、前馈神经网络、残差连接、层归一化等操作；
*   最终返回编码器最后一层的输出和每一层的注意力权重矩阵。

### 3.1 Encoder Layer：单个编码器层

作为 **Encoder** 的组成单元, 每个 **Encoder Layer** 完成一次对输入的特征提取过程, 即编码过程。

结构如图所示：

![](https://img-blog.csdnimg.cn/ee33c946f81c4d86b619621d33cac196.png)

```
# ---------------------------------------------------#
# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
# ---------------------------------------------------#
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
 
    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model]，需要注意的是最初始的QKV矩阵是等同于这个
        输入的，去看一下enc_self_attn函数.
        """
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
```

> **Encoder Layer** 包含两个部分：**多头注意力机制**＋**前馈神经网络** 

### 3.2 Encoder：编码器

**Encoder** 用于对输入进行指定的特征提取过程，也称为编码，由 n 个 **Encoder Layer** 层堆叠而成。

结构如图所示：

![](https://img-blog.csdnimg.cn/4bca6c28864a41f4afaa49e7eb8c9ceb.png)

```
# -----------------------------------------------------------------------------#
# Encoder部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 这行其实就是生成一个矩阵，大小是: src_vocab_size * d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.pos_emb = PositionalEncoding(d_model)
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
 
    def forward(self, enc_inputs):
        """
        这里我们的enc_inputs形状是： [batch_size x source_len]
        """
        # 下面这行代码通过src_emb进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
 
        # 这行是位置编码，把两者相加放到了pos_emb函数里面
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
 
        # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # 去看EncoderLayer层函数
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

进入 Encoder 后，首先进行 Embedding，然后进行 Positional Encoding。Embedding 使用了 nn.Embedding。n 个 Encoder Layer 存放在 nn.ModuleList() 里的列表中。 

> **Encoder** 部分包含三个部分：**Word Embedding**＋**Position Embedding**＋**Multi-Head Attention 层及后续的 Feed Forward 层**

*   **Multi-Head Attention 层：** 主要就是进行 attention 的计算，QKV 的矩阵运算都在这里。
*   **Feed Forward 层：**  就是进行特征的提取，进行向前传播。

### 3.3 Padding Mask：形成一个符号矩阵

```
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    # 最终得到的应该是一个最后n列为1的矩阵，即K的最后n个token为PAD。
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
```

> **padding mask 的作用：**
> 
> 不同 batch 之间句子长度可以不一样，但是每个 batch 的长度必须是一样的：因此出现一个问题，**不够长度需要加 pad**，使得其长度变成一样。
> 
> 我们看一下这个图：
> 
> ![](https://img-blog.csdnimg.cn/f27b11df2e824f509d3bc058976118f9.png)
> 
> 阴影部分是没有意义的，所以我们希望它是 0，以便后续的 softmax 等操作。
> 
> padding mask 的主要作用就是**针对句子不够长的问题，我们加了 pad，因此需要对 pad 进行 遮掩 mask**。

**从代码角度来看：**

这个函数最核心的一句代码是 **seq_k.data.eq(0)**，这句的作用是**返回一个大小和 seq_k 一样的 tensor**，只不过里面的值只有 True 和 False。如果 seq_k 某个位置的值等于 0，那么对应位置就是 True，否则即为 False。

举个栗子：输入为 seq_data = [1, 2, 3, 4, 0]，seq_data.data.eq(0) 就会返回 [False, False, False, False, True]

**【注意】**由于在 Encoder 和 [Decoder](https://so.csdn.net/so/search?q=Decoder&spm=1001.2101.3001.7020) 中都需要进行 mask（和矩阵原大小一样，有问题的地方加负无穷） 操作，因此就无法确定这个函数的参数中 seq_len 的值，如果是在 Encoder 中调用的，seq_len 就等于 src_len；如果是在 Decoder 中调用的，seq_len 就有可能等于 src_len，也有可能等于 tgt_len（因为 Decoder 有两次 mask）。

🚀4. 解码器（Decoder）
-----------------

![](https://img-blog.csdnimg.cn/c86fc9844ca74465bd28b465a55cf579.png)

上图红色框框为 Transformer 的 **Decoder** 结构，与 **Encoder** 相似，但是存在一些区别。

**Decoder** 包含两个 **Multi-Head Attention** 层。

*   第一个 Multi-Head Attention 层采用了 **Masked** 操作。
*   第二个 Multi-Head Attention 层的 **K, V 矩阵**使用 **Encoder** 的**编码信息矩阵 C** 进行计算，而 **Q 使用上一个 Decoder 的输出计算**。

最后有一个 Softmax 层计算下一个翻译单词的概率。

### 4.1Decoder Layer：单个解码层 

**Decoder** 模块由 6 个 **Decoder Layer** 组成，每个 Decoder Layer 结构完全一样，如图所示：

![](https://img-blog.csdnimg.cn/d15023a7e26d4424a7fb61083249d03c.png)

```
# -----------------------------------------------------------------------------#
# Decoder Layer包含了三个部分：解码器自注意力、“编码器-解码器”注意力、基于位置的前馈网络
# -----------------------------------------------------------------------------#
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
 
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
```

> **Decoder Layer** 包含了三个部分：**解码器自注意力 +“编码器 - 解码器” 注意力 + 基于位置的前馈网络**

每个 **Decoder Layer** 由三个子层连接结构组成：

*   **第一个子层连接结构：**包括一个 **Multi-Head Attention** 和 **Norm 层**以及一个**残差连接**。

在训练时，因为有目标数据可用，所以第一个 **Decoder Layer** 的 **Multi-Head Attention** 的输入来自于目标数据，但是在测试时，已经没有目标数据可用了，那么，输入数据就来自于此前序列的 **Decoder** 模块输出，没有预测过，那么就是起始标志的编码。同时，**这里的注意力是自注意力**，也就是说 **Q、K、V 都来自于目标数据矩阵变化得来**，然后计算注意力，另外，这里计算注意力值时，一定使用 Mask 操作。后续的 5 个 **Decoder Layer** 的输入数据是前一个 **Decoder Layer** 的输出。

*    **第二个子层连接结构：**包括一个 **Multi-Head Attention** 和 **Norm 层**以及一个**残差连接**。

**Encoder** 的输出的结果将会作为 **K、V** 传入每一个 **Decoder Layer** 的第二个子层连接结构，而 **Q** 则是当前 **Decoder Layer** 的上一个子层连接结构的输出。注意，**这里的 Q、K、V 已经不同源了，所以不再是自注意力机制**。完成计算后，输出结果作为第三个子层连接结构的输入。

*   **第三个子层连接结构：** 包括一个**前馈全连接子层**和 **Norm 层**以及一个**残差连接**。

完成计算后，输出结果作为输入进入下一个 **Decoder Layer**。如果是最后一个 **Decoder Layer**，那么输出结果就传入输出模块。

### 4.2Decoder：解码器

```
# -----------------------------------------------------------------------------#
# Decoder 部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
 
    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, tgt_len, d_model]
 
        ## get_attn_pad_mask 自注意力层的时候的pad 部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
 
        ## get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
 
        ## 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
 
        ## 这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
 
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```

> **Decoder** 部分包含三个部分：**Word Embedding**＋**Position Embedding**＋**Multi-Head Attention 层及后续的 Feed Forward 层**

**Decoder** 和 **Encoder** 类似，就是将 6 个 **Decoder Layer** 进行堆叠。第一个 **Decoder Layer** 接受目标数据作为输入，后续的 **Decoder** 使用前序一个 **Decoder Layer** 的输出作为输入，通过这种方式将 6 个 **Decoder Layer** 连接。最后一个 **Decoder Layer** 的输出将进入输出模块。 

### 4.3 Sequence Mask：屏蔽子序列的 mask

屏蔽子序列的 mask 部分，这个函数就是用来表示 **Decoder** 的输入中哪些是未来词，使用一个上三角为 1 的矩阵遮蔽未来词，让当前词看不到未来词。

![](https://img-blog.csdnimg.cn/img_convert/ac0b52b518828a0bbb67ce1db5d8cd12.jpeg)

```
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]
```

🚀5. 位置编码（Position Embedding）
-----------------------------

Transformer 中需要使用 Position Embedding 表示单词出现在句子中的位置。因为 Transformer 不采用 RNN 的结构，而是使用全局信息，因此是无法捕捉到序列顺序信息的，例如将 K、V 按行进行打乱，那么 Attention 之后的结果是一样的。但是序列信息非常重要，代表着全局的结构，因此必须将序列的分词相对或者绝对 position 信息利用起来。  
 

![](https://img-blog.csdnimg.cn/9102c8a065624f86a9b1f44b56d01a2f.png)

```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
 
        self.dropout = nn.Dropout(p=dropout)
		# 生成一个形状为[max_len,d_model]的全为0的tensor
        pe = torch.zeros(max_len, d_model)
        # position:[max_len,1]，即[5000,1]，这里插入一个维度是为了后面能够进行广播机制然后和div_term直接相乘
        # 注意，要理解一下这里position的维度。每个pos都需要512个编码。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 共有项，利用指数函数e和对数函数log取下来，方便计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
 
        # 这里position * div_term有广播机制，因为div_term的形状为[d_model/2],即[256],符合广播条件，广播后两个tensor经过复制，形状都会变成[5000,256]，*表示两个tensor对应位置处的两个元素相乘
        # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置赋值给pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # 同理，这里是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        # 上面代码获取之后得到的pe:[max_len*d_model]
 
        # 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
		# 定一个缓冲区，其实简单理解为这个参数不更新就可以，但是参数仍然作为模型的参数保存
        self.register_buffer('pe', pe)  
 
    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # 这里的self.pe是从缓冲区里拿的
        # 切片操作，把pe第一维的前seq_len个tensor和x相加，其他维度不变
        # 这里其实也有广播机制，pe:[max_len,1,d_model]，第二维大小为1，会自动扩张到batch_size大小。
        # 实现词嵌入和位置编码的线性相加
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

位置编码的实现直接对照着公式写就行，上面这个代码只是其中一种实现方式。  
**【注意】**pos 代表的是单词在句子中的绝对索引位置，例如 max_len 是 128，那么索引就是从 0,1,2,…,127，假设 d_model 是 512，即用一个 512 维 tensor 来编码一个索引位置，那么 0<=2i<512，0<=i<=255，那么 2i 对应取值就是 0,2,4…510，即偶数位置；2i+1 的取值是 1,3,5…511，即奇数位置。

最后的文本嵌入表征是词嵌入和位置编码相加得到。

🚀6. 注意力机制（Attention）
---------------------

### 6.1Scaled DotProduct Attention：缩放点积注意力机制

![](https://img-blog.csdnimg.cn/0055d8111cb44ebe823d83396237cf4d.png)

```
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
 
    def forward(self, Q, K, V, attn_mask):
        # 输入进来的维度分别是Q:[batch_size x n_heads x len_q x d_k]  K:[batch_size x n_heads x len_k x d_k]  V:[batch_size x n_heads x len_k x d_v]
        # matmul操作即矩阵相乘
        # [batch_size x n_heads x len_q x d_k] matmul [batch_size x n_heads x d_k x len_k] -> [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
 
        # masked_fill_(mask,value)这个函数，用value填充源向量中与mask中值为1位置相对应的元素，
        # 要求mask和要填充的源向量形状需一致
        # 把被mask的地方置为无穷小，softmax之后会趋近于0，Q会忽视这部分的权重
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context:[batch_size,n_heads,len_q,d_k]
        # attn:[batch_size,n_heads,len_q,len_k]
        return context, attn
```

缩放点积注意力机制主要原理就是通过 **Q** 、**K** 计算出 **scores**，然后将 **scores** 和 **V 进行 matmul 操作，**即矩阵相乘，这样得到每个单词的 context vector。

首先将 **Q** 和 **K 的转置**相乘，**相乘之后得到的 scores 还不能立刻进行 softmax，需要和 attn_mask 相加，**把一些需要屏蔽的信息屏蔽掉，**attn_mask** 是一个仅由 True 和 False 组成的 tensor，并且一定会保证 **attn_mask** 和 **scores** 的维度四个值相同（不然无法做对应位置相加）

mask 完了之后，就可以对 **scores** 进行 softmax 了。然后再与 V 相乘，得到 **context。**

### 6.2MultiHead Attention：多头注意力机制

与其只使用单独一个注意力汇聚， 我们可以用独立学习得到的 h 组（一般 h=8）不同的线性投影来变换 Q、K 和 V。

然后，这 h 组变换后的 Q、K 和 V 将并行地送到注意力汇聚中。 最后，将这 h 个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换， 以产生最终输出。 这种设计被称为**多头注意力（multihead attention）。**

![](https://img-blog.csdnimg.cn/c3a8420693c641bb80be1f96b321dec8.png)

```
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Wq,Wk,Wv其实就是一个线性层，用来将输入映射为Q、K、V
        # 这里输出是d_k * n_heads，因为是先映射，后分头。
        self.W_Q = nn.Linear(d_model, d_k * n_heads) 
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
 
    def forward(self, Q, K, V, attn_mask):
        # attn_mask:[batch_size,len_q,len_k]
        # 输入的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], 
        # V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
 
        # 分头；一定要注意的是q和k分头之后维度是一致的，所以一看这里都是d_k
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
 
        # attn_mask:[batch_size x len_q x len_k] ---> [batch_size x n_heads x len_q x len_k]
        # 就是把pad信息复制n份，重复到n个头上以便计算多头注意力机制
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
 
        # 计算ScaledDotProductAttention
        # 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v],
        # attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # 这里实际上在拼接n个头，把n个头的加权注意力输出拼接，然后过一个线性层，context变成
        # [batch_size,len_q,n_heads*d_v]。这里context需要进行contiguous，因为transpose后源tensor变成不连续的
        # 了，view操作需要连续的tensor。
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        # 过残差、LN，输出output: [batch_size x len_q x d_model]和这一层的加权注意力表征向量
        return self.layer_norm(output + residual), attn
```

代码中有三处地方调用 **MultiHeadAttention()：**

*   **Encoder Layer** 调用一次，传入的 **input_Q**、**input_K**、**input_V** 全部都是 **enc_inputs**；
*   **Decoder Layer** 中两次调用：
    *   第一次传入的全是 **dec_inputs**，
    *   第二次传入的分别是 **dec_outputs，enc_outputs，enc_outputs**

> 这里需要注意一下：为啥都是 d_k 而不是 d_q 呢？![](https://img-blog.csdnimg.cn/11499ed5b5e84576935026915543e802.png)
> 
> 我们要注意的是 q 和 k 分头之后维度是一致的，所以这里都是 dk

🚀7. 前馈神经网络（PoswiseFeedForward）
-------------------------------

完成多头注意力计算后，考虑到此前一系列操作对复杂过程的拟合程度可能不足，所以通过增加全连接层来增强模型的拟合能力。 

有两种实现方式：一种是通过卷积的方式实现，一种是通过线性层实现。二者的区别除了原理上，还有代码细节上。

举个栗子：

*   第一种卷积方式实现要求输入必须是 **[batch_size,channel,length]**，必须是**三维 tensor**
*   第二种线性层方式实现要求输入是 **[batch_size,*,d_model]**，可以有**多个维度**

#### 7.1 实现方式 1：Conv1d

```
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
 
    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
```

###  **7.2 实现方式 2：Linear**

```
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        
    def forward(self, inputs):              # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).(output + residual)   # [batch_size, seq_len, d_model]
```

这个方式比较好理解，就是做两次线性变换，残差连接后再跟一个 Layer Norm

> **Layer Norm 的作用：**对 x 归一化，使 x 的均值为 0，方差为 1

以上就是 transformer 代码的解读。

更多详细解读还要看各位大佬的：

b 站：[Transformer 代码 (源码 Pytorch 版本) 从零解读(Pytorch 版本）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dR4y1E7aL?p=1&vd_source=84352d0824b330075f0cb61978ba4fbb "Transformer代码(源码Pytorch版本)从零解读(Pytorch版本）_哔哩哔哩_bilibili")

 [手把手教你用 Pytorch 代码实现 Transformer 模型_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1vf4y1n7k2/?spm_id_from=333.788.recommend_more_video.1&vd_source=84352d0824b330075f0cb61978ba4fbb "         手把手教你用Pytorch代码实现Transformer模型_哔哩哔哩_bilibili")

CSDN：[Transformer 的 PyTorch 实现（超详细）](https://mathor.blog.csdn.net/article/details/107352273?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-107352273-blog-126187598.235%5Ev31%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-107352273-blog-126187598.235%5Ev31%5Epc_relevant_default_base&utm_relevant_index=2 "Transformer的PyTorch实现（超详细）") 

![](https://img-blog.csdnimg.cn/e012ccba0b5d48a59e42ecf4756cfb91.gif)