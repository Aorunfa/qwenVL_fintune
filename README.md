# qwenVL_fintune
一个微调qwenVL系列的仓库。总结qwenVL系列差异，对比llval。

## qwenVL

模型设计上本质与llava类似，使用一个vit作为vision encoder，使用一个训练好llm作为vison-text decoder，中间增加一个adapter将vison token与text token进行聚合

vison encoder使用vit结构，权重初始化石红clip模型。llm使用qwen系列。

adapter (Position-aware Vision-Language Adapter) 由一个cross-attention层构成，使用一个长度固定embedding矩阵作为query，key和value由vison token得到，查询最终得到固定长度的vison feature token。这其中需要注意的是：
    * query是可学习参数
    * vison feature token的sequence长度固定
    * attention操作增加三角位置编码，标记位置信息

image patch process将image resize到固定的分辨率，e.g 224×224，patch size设置为14提取vison token

训练包含两个预训练和一个指令微调阶段，qwenVL相较于llava的优势在于: 文档图片解析与理解、目标定位、多图对比能力
* stage 1 对齐预训练：冻结llm全量训练vision-encoder和adapter, 输入图片resize到224×224

* stage 2 多任务预训练：全量训练整个模型, 输入图片resize到448×448。增加OCR、caption、VQA、Grounding等各类任务混合数据

* stage 3 指令微调：冻结vison-encoder全量微调adaptor和llm。混合多模态和纯文本对话数据，多模态数据增加位置信息相关的数据，纯文本数据用于保持模型的对话能力

## qwenVL2
在qwenVL的基础上增加支持输入图片任意分辨率提取相应数量的vision token。对于token数量变化，沿用绝对位置编码标记位置信息变得困难，改进使用相对位置编码。

朴素动态分辨率机制：捕捉相对位置，减少token数量
    * 相对位置编码： vision encoder使用2D-ROPE进行位置编码
    * 相邻token特征合并：对提取得到的visual token特征还原位置关系，使用一个MLP层对相邻的2×2的token feature合并为一个token
    * e.g 224×224图片patch_size为14，可以提取num_vison_token = 224 / 14 / 2 * 224 / 14 / 2 = 64，最后前后增加一个标志token，合计66个

多模态旋转位置编码：embedding的位置编码将ROPE解构成三个维度，时间，空间height，空间width。位置标记序号(t, h, w)，t用于定位视频帧号，(h, w)用于定位帧的patch位置。文本的三个坐标相同，取上一个模态的max(t, h, w)。

统一视频和图片理解：使用深度为2的3D卷积提取patch，视频每秒取两帧，为了统一，图片需要copy为两张。动态调整视频帧分辨率，是的最终token数量低于最大值阈值。

训练阶段和qwenVL相同，区别在于数量量级和多任务数据集类型，如机器界面操作指令数据。

待定疑问：adapter是否沿用qwenVL，沿用的话adapter的query长度如何变化。
    推测：沿用，对应的query的sequence长度与vision token对应

## qwenVL2-5



## qwenVL系列与Llava对比


# 微调实战
参考仓库