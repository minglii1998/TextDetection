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

### 3.1 多通道图像卷积基础知识介绍

多通道图像+多卷积核做卷积

![](https://pic1.zhimg.com/80/v2-8d72777321cbf1336b79d839b6c7f9fc_hd.jpg)

输入有3个通道，同时有2个卷积核。对于每个卷积核，先在输入3个通道分别作卷积，再将3个通道结果加起来得到卷积输出。所以对于某个卷积层，无论输入图像有多少个通道，输出图像通道数总是等于卷积核数量。

### 3.2 anchors

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

那么这9个anchors是做什么的呢？借用Faster RCNN论文中的原图，遍历Conv layers计算获得的feature maps，为每一个点都配备这9种anchors作为初始的检测框。

![](https://pic1.zhimg.com/80/v2-c93db71cc8f4f4fd8cfb4ef2e2cef4f4_hd.jpg)

在原文中使用的是ZF model中，其Conv Layers中最后的conv5层num_output=256，对应生成256张特征图，所以相当于feature map每个点都是256-dimensions

在conv5之后，做了rpn_conv/3x3卷积且num_output=256，相当于每个点又融合了周围3x3的空间信息（猜测这样做也许更鲁棒？），同时256-d不变

假设在conv5 feature map中每个点上有k个anchor（默认k=9），而每个anhcor要分positive和negative，所以每个点由256d feature转化为cls=2k scores；而每个anchor都有(x, y, w, h)对应4个偏移量，所以reg=4k coordinates

补充一点，全部anchors拿去训练太多了，训练程序会在合适的anchors中随机选取128个postive anchors+128个negative anchors进行训练（什么是合适的anchors下文有解释）

![](https://pic2.zhimg.com/80/v2-4b15828dfee19be726835b671748cc4d_hd.jpg)

其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！

那么Anchor一共有多少个？原图800x600，VGG下采样16倍，feature map每个点设置9个Anchor，所以：

ceil(800/16)*ceil(600/16)*9=17100

其中ceil()表示向上取整，是因为VGG输出的feature map size= 50*38。

### 3.3 softmax判定positive与negative

一副MxN大小的矩阵送入Faster RCNN网络后，到RPN网络变为(M/16)x(N/16)，不妨设 W=M/16，H=N/16。在进入reshape与softmax之前，先做了1x1卷积，如图：

![](https://pic4.zhimg.com/80/v2-1ab4b6c3dd607a5035b5203c76b078f3_hd.jpg)

可以看到其num_output=18，也就是经过该卷积的输出图像为WxHx18大小。这也就刚好对应了feature maps每一个点都有9个anchors，同时每个anchors又有可能是positive和negative，所有这些信息都保存W*H*(9*2)大小的矩阵。为何这样做？后面接softmax分类获得positive anchors，也就相当于初步提取了检测目标候选区域box（一般认为目标在positive anchors中）。

加一个reshape是为了单独腾出一个维度方便softmax函数进行分类并保存，然后再接一个reshape恢复。

### 3.4 bounding box regression原理

如图所示绿色框为飞机的Ground Truth(GT)，红色为提取的positive anchors，即便红色的框被分类器识别为飞机，但是由于红色的框定位不准，这张图相当于没有正确的检测出飞机。所以我们希望采用一种方法对红色的框进行微调，使得positive anchors和GT更加接近。

![](https://pic4.zhimg.com/80/v2-93021a3c03d66456150efa1da95416d3_hd.jpg)

红色的框A代表原始的ositive Anchors，绿色的框G代表目标的GT，我们的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'

![](https://pic2.zhimg.com/80/v2-ea7e6e48662bfa68ec73bdf32f36bb85_hd.jpg)

先做平移，再做拉伸，当输入的anchor A与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调。

（这部分具体看原文吧）

### 3.5 对proposals进行bounding box regression

在了解bounding box regression后，再回头来看RPN网络第二条线路，如图

![](https://pic3.zhimg.com/80/v2-8241c8076d60156248916fe2f1a5674a_hd.jpg)

可以看到其 num_output=36，即经过该卷积输出图像为WxHx36，在caffe blob存储为[1, 4x9, H, W]，这里相当于feature maps每个点都有9个anchors，每个anchors又都有4个用于回归的变换量。

VGG输出 50*38*512 的特征，对应设置 50*38*k 个anchors，而RPN输出：

大小为 50*38*2*k 的positive/negative softmax分类特征矩阵
大小为 50*38*4*k 的regression坐标回归特征矩阵

恰好满足RPN完成positive/negative分类+bounding box regression坐标回归

### 3.6 Proposal Layer

Proposal Layer负责综合所有变换量和positive anchors，计算出精准的proposal，送入后续RoI Pooling Layer。

Proposal Layer forward（caffe layer的前传函数）按照以下顺序依次处理：

1. 生成anchors，利用[dx(A),dy(A),dw(A),dh(A)]对所有的anchors做bbox regression回归（这里的anchors生成和训练时完全一致）

2. 按照输入的positive softmax scores由大到小排序anchors，提取前pre_nms_topN(e.g. 6000)个anchors，即提取修正位置后的positive anchors。

3. 限定超出图像边界的positive anchors为图像边界（防止后续roi pooling时proposal超出图像边界）

4. 剔除非常小（width<threshold or height<threshold）的positive anchors

5. 进行nonmaximum suppression

6. Proposal Layer有3个输入：positive和negative anchors分类器结果rpn_cls_prob_reshape，对应的bbox reg的（eg. 300）结果作为proposal输出。

7. 之后输出proposal=[x1, y1, x2, y2]，注意，由于在第三步中将anchors映射回原图判断是否超出边界，所以这里输出的proposal是对应MxN输入图像尺度的，这点在后续网络中有用。另外我认为，严格意义上的检测应该到此就结束了，后续部分应该属于识别了。

RPN网络结构就介绍到这里，总结起来就是：

生成anchors -> softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals

## 4 RoI pooling

### 4.1 为何需要RoI Pooling

先来看一个问题：对于传统的CNN（如AlexNet，VGG），当网络训练好后输入的图像尺寸必须是固定值，同时网络输出也是固定大小的vector or matrix。如果输入图像大小不定，这个问题就变得比较麻烦。

有2种解决办法：

1. 从图像中crop一部分传入网络

2. 将图像warp成需要的大小后传入网络

无论采取那种办法都不好，要么crop后破坏了图像的完整结构，要么warp破坏了图像原始形状信息。

### 4.2 RoI Pooling原理

1. 由于proposal是对应 M*N 尺度的，所以首先使用spatial_scale参数将其映射回 （M/16)*(N/16) 大小的feature map尺度；

2. 再将每个proposal对应的feature map区域水平分为 pooled_w * pooled_h 的网格；

3. 对网格的每一份都进行max pooling处理。

这样处理后，即使大小不同的proposal输出结果都是 pooled_w * pooled_h 固定大小，实现了固定长度输出。

## 5 Classification

Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。

![](https://pic2.zhimg.com/80/v2-9377a45dc8393d546b7b52a491414ded_hd.jpg)

从PoI Pooling获取到7x7=49大小的proposal feature maps后，送入后续网络，可以看到做了如下2件事：

1. 通过全连接和softmax对proposals进行分类，这实际上已经是识别的范畴了
2. 再次对proposals进行bounding box regression，获取更高精度的rect box

## 6 Faster R-CNN训练

好像具体的训练也挺复杂的，但是毕竟不用跑，就不想看了...而且谁知道pytorch的跑法和caffe一样不一样呢？