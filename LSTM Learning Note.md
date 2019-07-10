LSTM Learning Notes
=====

参考：

[人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)

[[译] 理解 LSTM 网络](https://www.jianshu.com/p/9dc9f41f0b29)

## 1 RNN

循环神经网络（Recurrent Neural Network，RNN）是一种用于处理序列数据的神经网络。相比一般的神经网络来说，他能够处理序列变化的数据。

### 1.1 普通RNN

![](https://pic4.zhimg.com/80/v2-f716c816d46792b867a6815c278f11cb_hd.jpg)

x 为当前状态下数据的输入， h 表示接收到的上一个节点的输入。

y 为当前节点状态下的输出，而  h' 为传递到下一个节点的输出。

通过上图的公式可以看到，输出 h' 与 x 和 h 的值都相关。

![](https://pic2.zhimg.com/80/v2-71652d6a1eee9def631c18ea5e3c7605_hd.jpg)

### 1.2 RNN 问题

有时候，我们仅仅需要知道先前的信息来执行当前的任务。例如，我们有一个语言模型用来基于先前的词来预测下一个词。如果我们试着预测 “the clouds are in the sky” 最后的词，我们并不需要任何其他的上下文 —— 因此下一个词很显然就应该是 sky。在这样的场景中，相关的信息和预测的词位置之间的间隔是非常小的，RNN 可以学会使用先前的信息。

但是同样会有一些更加复杂的场景。假设我们试着去预测“I grew up in France... I speak fluent French”最后的词。当前的信息建议下一个词可能是一种语言的名字，但是如果我们需要弄清楚是什么语言，我们是需要先前提到的离当前位置很远的 France 的上下文的。这说明相关信息和当前预测位置之间的间隔就肯定变得相当的大。

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。

![](https://upload-images.jianshu.io/upload_images/42741-9ac355076444b66f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

## 2 LSTM

### 2.1 简介

长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现，可以学习长期依赖信息。

![](https://pic4.zhimg.com/80/v2-e4f9851cad426dfe4ab1c76209546827_hd.jpg)

相比RNN只有一个传递状态  h<sup>t</sup> ，LSTM有两个传输状态，一个  c<sup>t</sup>  （cell state），和一个  h<sup>t</sup>  （hidden state）。

![](https://upload-images.jianshu.io/upload_images/42741-b9a16a53d58ca2b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

LSTM 的关键就是细胞状态，水平线在图上方贯穿运行。细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。

![](https://upload-images.jianshu.io/upload_images/42741-ac1eb618f37a9dea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

### 2.2 深入理解

![](https://pic4.zhimg.com/80/v2-15c5eb554f843ec492579c6d87e1497b_hd.jpg)
![](https://pic1.zhimg.com/80/v2-d044fd0087e1df5d2a1089b441db9970_hd.jpg)

其中， z<sup>f</sup> ， z<sup>i</sup> ，z<sup>o</sup> 是由拼接向量乘以权重矩阵之后，再通过一个 sigmooid 激活函数转换成0到1之间的数值，来作为一种门控状态。而  z 则是将结果通过一个 tanh 激活函数将转换成-1到1之间的值（这里使用 tanh 是因为这里是将其做为输入数据，而不是门控信号）。

![](https://pic2.zhimg.com/80/v2-556c74f0e025a47fea05dc0f76ea775d_hd.jpg)

LSTM内部主要有三个阶段：

1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行选择性忘记。简单来说就是会 “忘记不重要的，记住重要的”。

   具体来说是通过计算得到的 z<sup>f</sup> （f表示forget）来作为忘记门控，来控制上一个状态的 c<sup>t-1</sup> 哪些需要留哪些需要忘。

2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入  x<sup>t</sup> 进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些。当前的输入内容由前面计算得到的 z 表示。而选择的门控信号则是由 z<sup>i</sup> （i代表information）来进行控制。

   将上面两步得到的结果相加，即可得到传输给下一个状态的 c<sup>t</sup> 。也就是上图中的第一个公式。

3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过  z<sup>o</sup> 来进行控制的。并且还对上一阶段得到的 c<sup>o</sup> 进行了放缩（通过一个tanh激活函数进行变化）。

   与普通RNN类似，输出 y<sup>t</sup> 往往最终也是通过 h<sup>t</sup> 变化得到。

### 2.3 LSTM 变体

其中一个变体是流形的 LSTM 变体，增加了 “peephole connection”。是说，我们让 门层 也会接受细胞状态的输入。

![](https://upload-images.jianshu.io/upload_images/42741-0f80ad5540ea27f9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

另一个变体是通过使用 coupled 忘记和输入门。不同于之前是分开确定什么忘记和需要添加什么新的信息，这里是一同做出决定。我们仅仅会当我们将要输入在当前位置时忘记。我们仅仅输入新的值到那些我们已经忘记旧的信息的那些状态 

![](https://upload-images.jianshu.io/upload_images/42741-bd2f1feaea22630e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

另一个改动较大的变体是 Gated Recurrent Unit (GRU)。它将忘记门和输入门合成了一个单一的 更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。

![](https://upload-images.jianshu.io/upload_images/42741-dd3d241fa44a71c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

