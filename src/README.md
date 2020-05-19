# src

## VGG16

論文：https://arxiv.org/abs/1409.1556

### 説明

論文中の Table 1 に記載されている D を実装した.

当初は Batch Normalization, He の初期化が知られていなかったため,
deep なネットワークの学習に苦戦していた.

著者らはまず 4 層の CNN と 3 層の全結合層を持つ shallow なモデルをランダムに初期化した重みで訓練し, 学習済みの重みの間に新たな CNN を追加する 工夫を施すことで deep なモデルの構築に成功している.

本実装では Batch Normalization, He の初期化を採用しているため上記の工夫は実施していないが, 訓練が可能になっている.

### 結果

to be released.

## ResNet50
