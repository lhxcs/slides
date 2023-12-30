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
  - 对几何结构的依赖

<!--v-->

## Example

<div class="three-line">

|表头 a|表头 b|表头 c|
|:--:|:--:|:--:|
|这是一个|一些内容|...|
|三线表|...|...|

</div>

|表头 a|表头 b|表头 c|
|:--:|:--:|:--:|
|这是一个|一些内容|...|
|普通表格|...|...|

<!--s-->

<div class="middle center">
<div style="width: 100%">

# Part 2. 基于神经表示的重建方案

</div>
</div>

<!--v-->

## NeuS

<div class="mul-cols">
<div class="col">

第一列

- list
- list

</div>

<div class="col">

第二列

```python
class MyClass:
    def __init__(self, ...):
        ...
    def method(self, ...):
        ...
```

</div>
</div>

<div class="mul-cols">
<div class="col">

第一列

- list
- list

</div>

<div class="col">

第二列

1. list 
2. list 
    - list

</div>

<div class="col">

第三列

```python
class MyClass:
    def __init__(self, ...):
        ...
    def method(self, ...):
        ...
```

</div>
</div>

<!--v-->

## Neuralangelo

<!--v-->



## LoD-NeuS

