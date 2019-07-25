Attention Machenism Learning Notes
============

[深度学习中Attention Mechanism详细介绍：原理、分类及应用](https://zhuanlan.zhihu.com/p/31547842)

Attention是一种用于提升基于RNN（LSTM或GRU）的Encoder + Decoder模型的效果的的机制（Mechanism），一般称为Attention Mechanism。Attention Mechanism目前非常流行，广泛应用于机器翻译、语音识别、图像标注（Image Caption）等很多领域，之所以它这么受欢迎，是因为Attention给模型赋予了区分辨别的能力，例如，在机器翻译、语音识别应用中，为句子中的每个词赋予不同的权重，使神经网络模型的学习变得更加灵活（soft），同时Attention本身可以做为一种对齐关系，解释翻译输入/输出句子之间的对齐关系，解释模型到底学到了什么知识，为我们打开深度学习的黑箱，提供了一个窗口

Attention Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销，这也是Attention Mechanism应用如此广泛的原因。

## 1. 原理

### 1.1 Attention Mechanism主要需要解决的问题

1、把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。

2、把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往是模型性能下降。

### 1.2 Attention Mechanism原理

传统的Seq2Seq结构：

![](https://pic3.zhimg.com/80/v2-2a5ba93492e047b60f4ebc73f2862fae_hd.jpg)

其中，Encoder把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：1、做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。2、做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（<EOS>）为止。

Attention Mechanism模块图解：

![](https://pic4.zhimg.com/80/v2-163c0c3dda50d1fe7a4f7a64ba728d27_hd.jpg)

（一顿代码乱秀没懂）

## 2.Attention Mechanism分类

### 2.1 soft Attention 和Hard Attention

Soft Attention是参数化的（Parameterization），因此可导，可以被嵌入到模型中去，直接训练。梯度可以经过Attention Mechanism模块，反向传播到模型其他部分。

相反，Hard Attention是一个随机的过程。Hard Attention不会选择整个encoder的输出做为其输入，Hard Attention会依概率Si来采样输入端的隐状态一部分来进行计算，而不是整个encoder的隐状态。为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法来估计模块的梯度。

两种Attention Mechanism都有各自的优势，但目前更多的研究和应用还是更倾向于使用Soft Attention，因为其可以直接求导，进行梯度反向传播。

### 2.2 Global Attention 和 Local Attention

Global Attention：传统的Attention model一样。所有的hidden state都被用于计算Context vector 的权重，即变长的对齐向量at，其长度等于encoder端输入句子的长度。


