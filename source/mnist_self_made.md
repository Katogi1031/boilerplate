---
title: 自作でMNISTを分類しよう
date: 2022-08-05
tags: ["fastai", "l1", "l2"]
excerpt: 平均絶対誤差と平均平方二乗誤差でMNISTの画像分類をしてみよう
---

# 数字のクラス分類器をつくろう
ここでは、ニューラルネットワークとかの実装ではなく、実際に手で理論だてて数字のクラス分類器を作っていましょう。  
fastaiのデータセットを使って、3と7の数字を分類してみよう。

クラス分類器を作るにあたって、以下のような手順で行っていこうと思う。  
今回使うのは、平均絶対誤差(以下、L1ノルム)と平均平方二乗誤差(以下、L2ノルム)を使ってやってみよう。

- それぞれの数字の全ての画像を重ね合わせて、平均を求める
- 個々の画像と上記で求めた平均の画像のL1ノルムとL2ノルムを求める
- 識別できているかを確認する

では、早速やってみようと思うが、L1ノルムとL2ノルムの違いを簡単に触れておこう。
### 平均絶対誤差(L1ノルム)
---
まず、求め方は(個々の画素値 - 平均の画素値)の絶対値の和の平均から算出します。  
数式にすると、  
$$ L1ノルム = \frac{1}{n}\ \sum_{i=1}^{n} |a_i - f_i|$$

### 平均二乗誤差(L2ノルム)
求め方は、(個々の画素値 - 平均画素値)の2乗の平均から算出します。  
数式にすると、  
$$ L2ノルム = \sqrt{\frac{1}{n}\ \sum_{i=1}^{n} (a_i - f_i)^2}$$
こうすることで、大きい誤差ではより大きいペナルティを与え、小さい誤差には寛容な値を出すことができます。

これで必要な知識がそろったので、実際にコーディングしていきましょう。

---
## 数字ごとの画像の平均を求める
---
まずは、fastaiをインポートします。
```python3
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.all import *
```
次に、MNISTのデータセットを取得します。
```python3
path = untar_data(URLs.MNIST_SAMPLE)
```
今回の使うのは3と7の数字だけなので、その画像を取得します。
```python3
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
```
試しに、3の画像を出力してみましょう。
```python3
im3_path = threes[1]
im3 = Image.open(im3_path)
im3
```
また、画像の画素値を見てみましょう。
```python3
im3_t = tensor(im3)
df = pd.DataFrame(im3_t[4:15, 4:22])
df.style.set_properties(**{'font-style':'6pt'}).background_gradient('Greys')
```
次に、2次元の画素値をテンソル型にしましょう。
```python3
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
```
そしたら、服すのテンソルを積み重ねて1つのテンソルにして、ついでに正則化しましょう。
```python3
stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float() / 255
```
また、テンソルのサイズを確認してみましょう。
```python3
stacked_threes.shape

>>> torch.Size([6131, 28, 28])
```
このことから、6131枚のがぞうがあり、それぞれ28×28ピクセルで構成されていることが分かります。  
では、この章最後に平均を求めましょう。といっても簡単です。  
全ての画像の画素値の平均を求めたいのでmean関数に0次元目(3であれば6131)を表す0を引数として代入します。
```python3
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)
```
では、平均の画像を出力してみましょう。
```python3
show_image(mean3)
```
ぼやけているように見えると思います。これが、全ての画像を積み重ねたときの平均値です。  
お疲れさまでした。

---
## L1ノルムとL2ノルムを求めよう
---


