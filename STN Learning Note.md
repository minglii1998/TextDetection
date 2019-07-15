STN Learning Notes
==========

参考：

[详细解读Spatial Transformer Networks（STN）-一篇文章让你完全理解STN了](https://blog.csdn.net/qq_39422642/article/details/78870629)

[Spatial Transformer Networks](https://zhuanlan.zhihu.com/p/37110107)

[Pytorch中的仿射变换(affine_grid)](https://www.jianshu.com/p/723af68beb2e)（这篇是我在搜affine_grid的时候找到的，没再讲原理而是讲如何实现的，值得一看）

## 1 STN 作用

关于平移不变性 ，对于CNN来说，如果移动一张图片中的物体，那应该是不太一样的。假设物体在图像的左上角，我们做卷积，采样都不会改变特征的位置，糟糕的事情在我们把特征平滑后后接入了全连接层，而全连接层本身并不具备 平移不变性 的特征。但是 CNN 有一个采样层，假设某个物体移动了很小的范围，经过采样后，它的输出可能和没有移动的时候是一样的，这是 CNN 可以有小范围的平移不变性 的原因。

Spatial Transformer Networks提出的空间网络变换层，具有平移不变性、旋转不变性及缩放不变性等强大的性能。这个网络可以加在现有的卷积网络中，提高分类的准确性。

如下图所示：输入手写字体，我们感兴趣的是黄色框中的包含数字的区域，那么在训练的过程中，学习到的空间变换网络会自动提取黄色框中的局部数据特征，并对框内的数据进行空间变换，得到输出output。综上所述，空间变换网络主要有如下三个作用：

1. 可以将输入转换为下一层期望的形式
2. 可以在训练的过程中自动选择感兴趣的区域特征
3. 可以实现对各种形变的数据进行空间变换

![](https://pic2.zhimg.com/80/v2-fb4f4935445f248574d4659b24ebff49_hd.jpg)

## 2 STN基本架构

![](https://img-blog.csdn.net/20171221204520193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如图是Spatial Transformer Networks的结构，主要的部分一共有三个，它们的功能和名称如下：

* 参数预测：Localisation net
* 坐标映射：Grid generator
* 像素的采集：Sampler

如下图是完成的一个平移的功能，这其实就是Spatial Transformer Networks要做一个工作。

![](https://img-blog.csdn.net/20171221163025631?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

假设是一个全连接层，n,m代表输出的值在输出矩阵中的下标，输入的值通过权值w，做一个组合，完成这样的变换。

## 3 Localisation net是如何实现参数的选取的

### 3.1 实现平移

![](https://img-blog.csdn.net/20171221170633240?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![](https://img-blog.csdn.net/20171221163549847?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

相当于就是单纯的每个后层的layer中的元素都依靠前层的所有数据乘以参数，这样参数量会非常大，是后层元素个数乘以前层元素个数，但是至少对于平移来说，会有很多参数是0.

### 3.2 实现缩放

如果要把图放大来看，在x→(X2)→x′, y→(X2)→y′将其同时乘以2，就达到了放大的效果了，用矩阵表示如下： 

![](https://img-blog.csdn.net/20171221192639934?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

缩小也是同样的原理，如果把这张图放到坐标轴来看，就是如图所示，加上偏执值0.5表示向右，向上同时移动0.5的距离，这就完成了缩小。

### 3.3 实现旋转

实现选中是通过三角函数实现的，具体的推导过程也不难，就不摘过来了，需要的话就直接进博客看吧。

可以简单的理解为cosθ,sinθ就是控制这样的方向的，把它当成权值参数，写成矩阵形式，就完成了旋转操作。 

![](https://img-blog.csdn.net/20171221193524098?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 3.4 实现剪切

剪切变换相当于将图片沿x和y两个方向拉伸，且x方向拉伸长度与y有关，y方向拉伸长度与x有关，用矩阵形式表示前切变换如下：

![](https://img-blog.csdn.net/20171222110137275?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这里没有给出具体的推导公式还是有点伤的。

## 4 Grid generator实现像素点坐标的对应关系

无论如何做旋转，缩放，平移，只用到六个参数就可以了。我们定义如图的一个坐标矩阵变换关系：

![](https://img-blog.csdn.net/20171222085414269?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzk0MjI2NDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

 ( x<sup>i</sup><sub>t</sub> , y<sup>i</sup><sub>t</sub> )是输出的目标图片的坐标，( x<sup>s</sup><sub>i</sub> , y<sup>s</sup><sub>i</sub> )是原图片的坐标，Aθ表示仿射关系。

但仔细一点，这有一个非常重要的知识点,千万别混淆，我们的坐标映射关系是：

从目标图片→原图片

## 5.Sampler实现坐标求解的可微性

### 5.1 小数坐标问题的提出

