CRNN + CTC Learning Notes
===========

参考：

[一文读懂CRNN+CTC文字识别](https://zhuanlan.zhihu.com/p/43534801)

## 1 CRNN网络

整个CRNN网络可以分为三个部分

![](https://pic3.zhimg.com/80/v2-7ed5c65fe79dce49f006a9171cc1a80e_hd.jpg)

假设输入图像大小为 32 * 100 * 3

* Convlutional Layers

   这里的卷积层就是一个普通的CNN网络，用于提取输入图像的Convolutional feature maps，即将大小为 （32,100,3） 的图像转换为 （1,25,512） 大小的卷积特征矩阵。

* Recurrent Layers

   这里的循环网络层是一个深层双向LSTM网络，在卷积特征的基础上继续提取文字序列特征。

   所谓深层RNN网络，是指超过两层的RNN网络。

单层双向RNN网络：

![](https://pic4.zhimg.com/80/v2-9f5125e0c99924d2febf25bafd019d6f_hd.jpg)

深层双向RNN网络：

![](https://pic3.zhimg.com/80/v2-c0132f0b748eb031c696dae3019a2d82_hd.jpg)

stack形深层双向RNN网络：

![](https://pic2.zhimg.com/80/v2-00861a152263cff8b94525d8b8945ee9_hd.jpg)

* 在CRNN中显然使用了第二种stack形深层双向结构。

   由于CNN输出的Feature map是（1,25,512）大小，所以对于RNN最大时间长度 T = 25 （即有25个时间输入，每个输入 xt 列向量有 D=512 ）。

* Transcription Layers

   将RNN输出做softmax后，为字符输出。

在上文给出的实现中，为了将特征输入到Recurrent Layers，做如下处理：

* 首先会将图像缩放到 32 * W * 3 大小

* 然后经过CNN后变为 1 * （W/4） * 512

* 接着针对LSTM，设置 T = W/4 ， D=512 ，即可将特征输入LSTM。

所以在处理输入图像的时候，建议在保持长宽比的情况下将高缩放到 32 ，这样能够尽量不破坏图像中的文本细节（当然也可以将输入图像缩放到固定宽度，但是这样由于破坏文本的形状，肯定会造成性能下降）。

`问题`

对于Recurrent Layers，如果使用常见的Softmax Loss，则每一列输出都需要对应一个字符元素。那么训练时候每张样本图片都需要标记出每个字符在图片中的位置，再通过CNN感受野对齐到Feature map的每一列获取该列输出对应的Label才能进行训练

![](https://pic2.zhimg.com/80/v2-5803de0cd9eb4e20f6a722e02b196809_hd.jpg)

在实际情况中，标记这种对齐样本非常困难（除了标记字符，还要标记每个字符的位置），工作量非常大。另外，由于每张样本的字符数量不同，字体样式不同，字体大小不同，导致每列输出并不一定能与每个字符一一对应。

所以CTC提出一种对不需要对齐的Loss计算方法，用于训练网络，被广泛应用于文本行识别和语音识别中。

## 2 Connectionist Temporal Classification(CTC)

整个CRNN的流程如图。先通过CNN提取文本图片的Feature map，然后将每一个channel作为 D = 512 的时间序列输入到LSTM中。

![](https://pic2.zhimg.com/80/v2-6e2120edda0684a2a654d0627ad13591_hd.jpg)

* CNN Feature map

Feature map的每一列作为一个时间片输入到LSTM中。设Feature map大小为 m * T （图中 m = 512 ，T = 25 ）。下文中的时间序列 t 都从 t = 1 开始，即 1<=t<=T 。

定义为：x = ( x<sup>1</sup> , x<sup>2</sup> ,..., x<sup>T</sup> )

每一列：x<sup>t</sup> = ( x<sub>1</sub><sup>t</sup> , x<sub>21</sub><sup>t</sup> ,..., x<sub>m</sub><sup>t</sup> )

* LSTM

LSTM的每一个时间片后接softmax，输出 [公式] 是一个后验概率矩阵，定义为：y = ( y<sup>1</sup> , y<sup>2</sup> ,..., y<sup>T</sup> )

每一列：y<sup>t</sup> = ( y<sub>1</sub><sup>t</sup> , y<sub>21</sub><sup>t</sup> ,..., y<sub>m</sub><sup>t</sup> )

### CTC 总结

CTC是一种Loss计算方法，用CTC代替Softmax Loss，训练样本无需对齐。

CTC特点：

* 引入blank字符，解决有些位置没有字符的问题

* 通过递推，快速计算梯度

## CRNN+CTC总结

这篇文章的核心，就是将CNN/LSTM/CTC三种方法结合：

* 首先CNN提取图像卷积特征

* 然后LSTM进一步提取图像卷积特征中的序列特征

* 最后引入CTC解决训练时字符无法对齐的问题