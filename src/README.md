# src

## VGG16

論文：https://arxiv.org/abs/1409.1556

### 説明

論文中の Table 1 に記載されている D を実装した.

論文では 4 層の CNN と 3 層の全結合層を持つモデルを 1 度つくり, このモデルの中間層に新たな CNN を追加していくことで deep なモデルを構築していた.

当時は勾配消失問題に対する研究が進んでいなかったことからこのような手段が必要だったが, 現在は Batch Normalization, He の初期化によって上記の工夫は不要になっている. 本実装でも Batch Normalization, He の初期化を採用した.

### 結果

to be released.

## ResNet50
