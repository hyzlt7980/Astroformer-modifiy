# Astroformer-modifiy | Astroformer的优化和提升

A innovative approach to modify astroformer: wisely global connectivity ? swin?  and enpowered standard se

Astroformer复现与优化: https://arxiv.org/pdf/2304.05350
Astroformer 是 TinyImageNet 的 SOTA 模型，其变体 Astroformer-5 在 TinyImageNet 上取得了 92.98% 的准确率。TinyImageNet 最初由斯坦福大学 CS231n 课程团队整理，作为连接“小型教学数据”和“工业级大规模数
据”之间的中等规模图像分类基准，既能考察模型设计，也兼顾计算资源可行性。

1.原版 Astroformer-5 在 300 个 epoch（将 64 的输入图像上采样到 224）训练后，达到了 92.38% 的准确率。模型由 CCCT 四个 Block 构成，其中 C Block 包含 1×1 通道
扩展、3×3 卷积和轻量级 SE，用于构建全局依赖。我在 100 个 epoch、原始 64 大小图像输入下复现，获得了 61% 的准确率.
分析1：
作者在网络最前端将原始 64×64 图像上采样至 224×224，虽未引入真实新细节，但通过将稀疏锯齿的像素映射到更细密的网格，使卷积和注意力模块能在更连续、更丰富的特征图上捕捉细粒度空间依赖，从而提升分类性能。

2. 原本的 C block(1x1 conv+3x3 conv+1x1 conv+ light weight se);
   New design:  (1x1 conv+3x3 conv+cross scan(UCAS,Huawei Inc ,cvpr)+1x1 conv+ standard se,并在最后一个 T Block（Transformer Block）中使用 CARAFE(CVPR) 将 4×4 Patch 上采样至 14×14， 以替代 Astroformer-5 初始阶段的上采样。但在 64 大小输入和 100 epoch 条件下，仅达到了 58% 的准确率，AI 分析推测是 SE破坏了 Cross Scan 构建的全局依赖。

分析2: 标准SE在“挤压”阶段仅用GlobalAvgPool对每个通道求平均，而Cross-Scan输出是峰值 + 低激活。平均池化会将这些峰值与大量低激活混合，生成的数值不能准确的代表该feature map所包含的信息，
最后生成的门控系数(2层neuron layer之后的输出)可能无法准确反映每个特征图的重要性,这会导致特征图之间的关系被错误地建立。再次输入全局average pooling然后FC->FC, 大概率无法修改这样的错误，再次生成错误的门控系数，这些门控系数乘以每个feature map, 再次错误的调整了
feature map之间的关系, 以上过程循环，错误会被重复累积，造成准确率的大幅度下降（3%）

解决方案：
2.1 无论feature map(特征图)之间是否存在显式的依赖关系，通过convolution操作提取每个feature map的卷积特征图，并将这些特征图输入MLP生成门控系数，能够更聪明的调整每个特征图的贡献。
如果特征图之间存在依赖关系，这种方式能够有效地发现并加以利用；即使没有依赖关系，它也能增强模型的灵活性，提升其对不同特征的关注，从而提升性能和精度
（我自己提出的，innovative idea, 类似的academic paper: Convolutional Block Attention Module

2.2 并行使用GlobalMaxPool+GlobalAvgPool，对每个通道同时进行最大池化和平均池化，将两者拼接[Max,Avg]后输入小型MLPSigmoid；输出的通道门控既能捕捉局部峰值，又兼顾整体激活趋势，有
效避免“峰值被稀释. 这块是部分的convolutional block attention module

2.3 去掉standard se。

分析：
只在最后使用carafe进行上采样并输入到transformer block, 永远无法替代在一开始就用224×224输入并在高分辨率上执行完整CCCT模块所能获取的那部分细节和空间依赖
最后，考虑用dilated attention替换cross scan, 或者用swin的W-MSA和SW-MSA来替换cross scan.
