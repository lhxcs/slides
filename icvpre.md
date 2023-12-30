---
title: 基于多视角图片的三维重建
separator: <!--s-->
verticalSeparator: <!--v-->
theme: simple
highlightTheme: github
css: custom.css
revealOptions:
    transition: 'slide'
    transitionSpeed: fast
    center: false
    slideNumber: "c/t"
    width: 1000
---

<div class="middle center">
<div style="width: 100%">

# 基于多视角图片的三维重建

<hr/>

By Hanxuan Li

</div>
</div>

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part 1.Multi-View Stereo



</div>
</div>

<!--v-->

## 算法原理
<div class="mul-cols">
<div class='col'>

- 基于多视角图片进行立体匹配，计算深度图。(PatchMatch算法)
  - Initialization
  - Propagation
  - Search
- 从深度图得到三维网格。
  - Poisson Reconstruction
  - Marching Cubes
- 纹理映射。
</div>

<img src='image/1.png' width=400 height=200>

</div>



<!--v-->

## 实现——COLMAP

```
$ colmap feature_extractor --database_path $DATASET_PATH/database.db --image_pat $DATASET_PATH/images

$ colmap exhaustive_matcher --database_path $DATASET_PATH/database.db

$ mkdir $DATASET_PATH/sparse

$ colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse

$ mkdir $DATASET_PATH/dense

$ colmap image_undistorter --image_path $DATASET_PATH/images --input_path $DATASET_PATH/sparse/0 --output_path $DATASET_PATH/dense --output_type COLMAP --max_image_size 2000

$ colmap patch_match_stereo --workspace_path $DATASET_PATH/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true

$ colmap stereo_fusion --workspace_path $DATASET_PATH/dense --workspace_format COLMAP --input_type geometric --output_path $DATASET_PATH/dense/fused.ply

$ colmap poisson_mesher --input_path $DATASET_PATH/dense/fused.ply --output_path $DATASET_PATH/dense/meshed-poisson.ply
```

<!--v-->

## 实验结果

- 总体效果符合预期，是COLMAP的正常水平。
- 耗时久。

<div align=center>
<img src='image/a.gif' width=500 height=300 >
</div>

- 总体轮廓清楚，衣服纹理清晰
- 无法重建面部
<!--v-->
## Summary of MVS

- 存在的问题
  - 光度一致性原理有时候并不满足
  - 对几何结构和纹理的依赖
  - 遮挡问题
  - 计算复杂度

- 需要表现更好的方法——深度学习
<div align=center>
<img src="https://th.bing.com/th/id/OIP.-tEyfNnxagm0wbXO96nvdAHaFw?rs=1&pid=ImgDetMain" width=400 height=300>
</div>
<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part 2. 基于神经表示的重建方案

</div>
</div>

<!--v-->

## Preliminaries

- 通过神经网络从多视角图像中学习场景的三维结构
- Advantages
  - 相比于MVS，能够处理更复杂的三维场景(几何，纹理等)
  - Advanced: 新视角合成，数据压缩，动态场景……

- Disadvantages:
  - 训练时长与计算资源
  - 泛化能力不足，一个模型代表一个场景


- 着重探讨基于三维信息存储结构(编码)对重建质量的影响

<!--v-->

## NeuS

将三维场景隐式存储在神经网络中

一些概念：
- SDF: 零等值面构成三维物体的表面 $f(x)=0$
  - 连续的方式来表示形状

- Volumn rendering: 
  - $C(r)=\int_{t_n}^{t_f}T(t)\sigma (\boldsymbol{r}(t))c(\boldsymbol{r}(t),\boldsymbol{d})dt, T(t)=exp(-\int_{t_n}^{t}\sigma(\boldsymbol{r}(s))ds)$
  - 由NeRF提出

<div align=center>
<img src='image/2.png'>
</div>
<!--v-->

## NeuS(continued)

- NeRF的局限
  - 任务是新视角合成，我们的任务是三维表面重建
  - 使用occupancy volumn进行场景表示，几何质量不高
- NeuS的方法：将体渲染应用于学习SDF的表示
  - 使用SDF符号距离场进行三维场景表示
  - 使用体渲染的方法：$C(o,v)=\int_0^{\infty}\omega (t)c(p(t),v)dt$
    - 其中权重$\omega$的性质
      - Unbiased
      - Occlusion-aware
  - 在损失函数中引入ekinoal item,保证表面光滑：
    - $L_{reg}=\frac{1}{mn}\sum_{k,l}(\Vert\nabla f(p_{k,i})\Vert_2-1)^2$

<!--v-->

## NeuS(experiments)

- 遇到的问题
  - 默认迭代30w次，但是算法5-10w次基本收敛
  - 得到稀疏点云模型后手动剔除不care的点云(otherwise 影响重建效果)
- 效果展示

<div align=center>
<figure class="third">
    <img src="image/3.png" width=200 height=200>
    <img src="image/4.png" width=200 height=200>
    <img src="image/5.png" width=200 height=200>
</figure>
</div>

<!--v-->
## Summary of NeuS
- Advantages
  - 在中小规模的重建下质量很高
  - 适应具有复杂结构和自遮挡的对象
- Strugglement of current methods
  - 耗时久，泛化性弱
  - 不能很好恢复真实场景的细节
  - 在超大场景重建下表现不佳(will be discussed)
    - 存储和计算资源的限制
    - 数据采样不均匀或不充分，局部or全局？


<!--v-->

## Neuralangelo——高保真神经表面重建
<div align=center>
<img src='image/6.png' width=25% height=25%>
</div>

以高保真度(很好地恢复细节)创造真实场景的虚拟复制品

- 重要的方法
  - 多分辨率哈希网格表示三维场景
  - 神经表面渲染SDF

- 使其实现的关键因素
  - 采用数值方法计算用于平滑操作的高阶导数
  - 在哈希网格上使用粗细优化策略以控制细节

<!--v-->
## Multi-resolution hash encoding

使用不同分辨率的网格编码三维点的空间信息

<img src='image/7.png' width=50% height=50%>

- 设置多分辨率的网格系统
  - coarse: 空间概率
  - fine: 表面细节
- 哈希映射
- 所有分辨率下的特性向量整合，输入到MLP，学习权重

<!--v-->

## Numerical Gradient Computation

Recap: SDF梯度场的ekinoal loss
- Problem: 哈希编码的导数仅在局部voxel连续
- 分析梯度改为数值梯度
  - 允许多个网格的哈希条目同时优化更新

<img src='image/8.png'>
<!--v-->

## Progressive Levels of Details

粗到细的优化策略

- 步长逐渐减小
  - 数值梯度的较大步长对应整体的轮廓
  - 较小的步长避免平滑细节

- 哈希网格分辨率
  - 最初仅启用粗糙哈希网格
  - 根据步长逐步激活更细的网格