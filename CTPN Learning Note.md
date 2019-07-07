CTPN Learning Notes
===========

参考：

[场景文字检测—CTPN原理与实现](https://zhuanlan.zhihu.com/p/34757009)

[CTPN - 自然场景文本检测](https://blog.csdn.net/zchang81/article/details/78873347)

### 1 CTPN 方法概括

1. 用VGG16的前5个Conv stage（到conv5）得到feature map(W*H*C)

2. 在Conv5的feature map的每个位置上取3*3*C的窗口的特征，这些特征将用于预测该位置k个anchor（anchor的定义和Faster RCNN类似）对应的类别信息，位置信息。

3. 将每一行的所有窗口对应的3*3*C的特征（W*3*3*C）输入到RNN（BLSTM）中，得到W*256的输出

4. 将RNN的W*256输入到512维的fc层

5. fc层特征输入到三个分类或者回归层中。

   第二个2k scores 表示的是k个anchor的类别信息（是字符或不是字符）。
   
   第一个2k vertical coordinate和第三个k side-refinement是用来回归k个anchor的位置信息。
   
   2k vertical coordinate表示的是bounding box的高度和中心的y轴坐标（可以决定上下边界），k个side-refinement表示的bounding box的水平平移量。
   
   这边注意，只用了3个参数表示回归的bounding box，因为这里默认了每个anchor的width是16，且不再变化（VGG16的conv5的stride是16）。回归出来的box如Fig.1中那些红色的细长矩形，它们的宽度是一定的。

6. 用简单的文本线构造算法，把分类得到的文字的proposal（图Fig.1（b）中的细长的矩形）合并成文本线

![](http://images2015.cnblogs.com/blog/1058268/201701/1058268-20170112203504978-1308296948.png)

### 2 CTPN网络结构

CTPN结构与Faster R-CNN基本类似，但是加入了LSTM层。假设输入N Images

1. 首先VGG提取特征，获得大小为 N*C*H*W 的conv5 feature map。

2. 之后在conv5上做 3*3 的滑动窗口，即每个点都结合周围 3*3 区域特征获得一个长度为 3*3*C 的特征向量。输出 N*9C*H*W 的feature map，该特征显然只有CNN学习到的空间特征。

3. 再将这个feature map进行Reshape
Reshape : N*9C*H*W → (NH)*W*9C

4. 然后以 Batch = NH 且最大时间长度 Tmax = W 的数据流输入双向LSTM，学习每一行的序列特征。双向LSTM输出 (NH)*W*256 ，再经Reshape恢复形状：
Reshape : (NH)*W*256 → N*256*H*W

   该特征既包含空间特征，也包含了LSTM学习到的序列特征。

5. 然后经过“FC”卷积层，变为 N*512*H*W 的特征

6. 最后经过类似Faster R-CNN的RPN网络，获得text proposals

conv5 feature map如何从 N*C*H*W 变为 N*9C*H*W ：

![](https://pic4.zhimg.com/80/v2-4399a8ecb012241fa542e084eb7d727f_hd.jpg)

在原版caffe代码中是用im2col提取每个点附近的9点临近点，然后每行都如此处理，而im2col是用于卷积加速的操作，即将卷积变为矩阵乘法，从而使用Blas库快速计算。到了tf，没有这种操作，所以一般是用conv2d代替im2col，即强行卷积 C → 9C

#### 2.1 为何使用双向LSTM

首先，为什么要使用LSTM？

CNN学习的是感受野内的空间信息，LSTM学习的是序列特征。对于文本序列检测，显然既需要CNN抽象空间特征，也需要序列特征（毕竟文字是连续的）。

双向LSTM实际上就是将2个方向相反的LSTM连起来，如图

![](https://pic1.zhimg.com/80/v2-bc5266c4587af49516adb2cee4351838_hd.jpg)

#### 2.2 如何通过"FC"卷积层输出产生图2-b中的Text proposals?

CTPN通过CNN和BLSTM学到一组“空间 + 序列”特征后，在"FC"卷积层后接入RPN网络。

这里的RPN与Faster R-CNN类似，分为两个分支：

* 左边分支用于bounding box regression。由于fc feature map每个点配备了10个Anchor，同时只回归中心y坐标与高度2个值，所以rpn_bboxp_red有20个channels

* 右边分支用于Softmax分类Anchor，具体RPN网络与Faster R-CNN完全一样，所以不再介绍，只分析不同之处。

![](https://pic2.zhimg.com/80/v2-8496528d21dfd1c4e90df4ff57fa6221_hd.jpg)

由于CTPN针对的是横向排列的文字检测，所以其采用了一组（10个）等宽度的Anchors，用于定位文字位置。Anchor宽高为：weights = [16]

heights = [11,16,23,33,48,68,97,139,198,283]

CTPN为fc feature map每一个点都配备10个上述Anchors。

![](https://pic2.zhimg.com/80/v2-93e22f54fb0231b3f763f2f8129913ad_hd.jpg)

这样设置Anchors是为了：

* 保证在 x 方向上，Anchor覆盖原图每个点且不相互重叠。

* 不同文本在 y 方向上高度差距很大，所以设置Anchors高度为11-283，用于覆盖不同高度的文本目标。

获得Anchor后，与Faster R-CNN类似，CTPN会做如下处理：

* Softmax判断Anchor中是否包含文本，即选出Softmax score大的正Anchor

* Bounding box regression修正包含文本的Anchor的中心y坐标与高度。

与Faster R-CNN不同的是，这里Bounding box regression不修正Anchor中心x坐标和宽度。

Anchor经过上述Softmax和 y 方向bounding box regeression处理后，会获得图所示的一组竖直条状text proposal。后续只需要将这些text proposal用文本线构造算法连接在一起即可获得文本位置。

![](https://pic1.zhimg.com/80/v2-447461eb54bcc3c93992ffd1c70bcfb8_hd.jpg)

#### 2.3 文本线构造算法

为了说明问题，假设某张图有图所示的2个text proposal，即蓝色和红色2组Anchor，CTPN采用如下算法构造文本线：

![](https://pic4.zhimg.com/80/v2-de8098e725d168a038f197ce0707faaf_hd.jpg)

1. 按照水平 x 坐标排序Anchor
2. 按照规则依次计算每个Anchor boxi 的 pair(boxj) ，组成 pair(boxi,boxj)
3. 通过 pair(boxi,boxj) 建立一个Connect graph，最终获得文本检测框

正向寻找：

1. 沿水平正方向，寻找和 boxi 水平距离小于50的候选Anchor
2. 从候选Anchor中，挑出与 boxi 水平方向 overlap > 7 的Anchor
3. 挑出符合条件2中Softmax score最大的 boxj

再反向寻找：

1. 沿水平负方向，寻找和 boxj 水平距离小于50的候选Anchor
2. 从候选Anchor中，挑出与 boxj 水平方向 overlap > 7 的Anchor
3. 挑出符合条件2中Softmax score最大的 boxk

最后对比 scorei 和 scorek :

* 如果 scorei >= scorek ，则这是一个最长连接，那么设置 Graph(i,j) = True
* 如果 scorei < scorek ，说明这不是一个最长的连接（即该连接肯定包含在另外一个更长的连接中）。

### 3 总结

1. 由于加入LSTM，所以CTPN对水平文字检测效果超级好。

2. 因为Anchor设定的原因，CTPN只能检测横向分布的文字，小幅改进加入水平Anchor即可检测竖直文字。但是由于框架限定，对不规则倾斜文字检测效果非常一般。

3. CTPN加入了双向LSTM学习文字的序列特征，有利于文字检测。但是引入LSTM后，在训练时很容易梯度爆炸，需要小心处理。

