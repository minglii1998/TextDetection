ResNet Learning Notes
==============

参考：

[一文简述ResNet及其多种变体](https://www.jiqizhixin.com/articles/042201)

### ResNet 简介

ResNet 的核心思想是引入一个所谓的「恒等快捷连接」（identity shortcut connection），直接跳过一个或多个层，如下图所示：

![](https://image.jiqizhixin.com/uploads/editor/4136d82a-72ea-4418-983f-c010a454f8f2/1524374872536.jpg)

对于一个堆积层结构（几层堆积而成）当输入为 x 时其学习到的特征记为 H(x) ，现在我们希望其可以学习到残差 F(x) = H(x) - x ，这样其实原始的学习特征是 F(x) + x 。之所以这样是因为残差学习相比原始特征直接学习更容易。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。

### ResNet 网络结构

ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。

ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。从图中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。

![](https://pic2.zhimg.com/80/v2-7cb9c03871ab1faa7ca23199ac403bd9_hd.jpg)

对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsamp，可以采用strde=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。

觉得..找了几篇关于ResNet的博客都讲得不是很清楚啊...感觉看完还是一知半解...令人窒息。
