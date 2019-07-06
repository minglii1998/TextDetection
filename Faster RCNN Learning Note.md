Faster RCNN Notes
==========

参考：

[一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)

## 1 整体架构

![](https://pic3.zhimg.com/80/v2-c0172be282021a1029f7b72b51079ffe_hd.jpg)

Faster RCNN可以分为4个主要内容：

* Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

* Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。

* Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

* Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

下图展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，可以清晰的看到该网络对于一副任意大小PxQ的图像，首先缩放至固定大小MxN，然后将MxN图像送入网络；而Conv layers中包含了13个conv层+13个relu层+4个pooling层；RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，然后计算出proposals；而Roi Pooling层则利用proposals从feature maps中提取proposal feature送入后续全连接和softmax网络作classification（即分类proposal到底是什么object）。

![](https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_hd.jpg)

## 2 Conv layers

Conv layers包含了conv，pooling，relu三种层。

在Conv layers中：

所有的conv层都是： kernal_size = 3, pad = 1, stride = 1

所有的pooling层都是： kernal_size = 2, pad = 0, stride = 2

在Faster RCNN Conv layers中对所有的卷积都做了扩边处理（ pad=1，即填充一圈0），导致原图变为 (M+2)x(N+2)大小，再做3x3卷积后输出MxN 。正是这种设置，导致Conv layers中的conv层不改变输入和输出矩阵大小。

一个MxN大小的矩阵经过Conv layers固定变为(M/16)x(N/16)！这样Conv layers生成的feature map中都可以和原图对应起来。

## 3 Region Proposal Networks(RPN)

Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，极大提升检测框的生成速度。

![](https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_hd.jpg)

可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。

而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。

其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

## 3.1 多通道图像卷积基础知识介绍

多通道图像+多卷积核做卷积

![](https://pic1.zhimg.com/80/v2-8d72777321cbf1336b79d839b6c7f9fc_hd.jpg)

输入有3个通道，同时有2个卷积核。对于每个卷积核，先在输入3个通道分别作卷积，再将3个通道结果加起来得到卷积输出。所以对于某个卷积层，无论输入图像有多少个通道，输出图像通道数总是等于卷积核数量。

## 3.2 anchors

anchors，实际上就是一组由rpn/generate_anchors.py生成的矩形。

```
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]
```

其中每行的4个值(x1,y1,x2,y2)表矩形左上和右下角点坐标。

9个矩形共有3种形状，长宽比为大约为width:height = {1:1,1:2,2:1}三种。实际上通过anchors就引入了检测中常用到的多尺度方法。

![](https://pic4.zhimg.com/80/v2-7abead97efcc46a3ee5b030a2151643f_hd.jpg)

