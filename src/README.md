# src

## VGG16

論文：https://arxiv.org/abs/1409.1556

### 説明

論文中の Table 1 に記載されている D を実装した.

論文では 4 層の CNN と 3 層の全結合層を持つモデルを 1 度つくり, このモデルの中間層に新たな CNN を追加していくことで deep なモデルを構築していた.

これは Batch Normalization, He の初期化に関する研究が発展しておらず勾配消失や勾配発散が問題となっていたためである。

本実装では Batch Normalization, He の初期化を採用しているため, end-to-end での学習が可能になっている.

### 結果

to be released.

## ResNet50
