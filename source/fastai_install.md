---
title: fastaiをインストールしよう
date: 2022-08-05
tags: ["fastai"]
excerpt: これは抜粋です。
---


# fastaiを使えるようになろう

ここでは、fastaiのインストールの仕方をGPUとCPUそれぞれ使う場合に分けて、解説していきます。
- Google ColabなどのクラウドでGPUを使う場合
- GPUを搭載していないPCのローカル環境で構築する場合

## [fastai](https://docs.fast.ai/)とは
> - ディープラーニングライブラリの一つで、使いやすさ、柔軟性、またはパフォーマンスに優れる。
> - そのためニューラルネットワークのトレーニングを簡潔に記述することができる。
> - GPUに最適化されており、高速に実行が可能。

### Googole Colabでfastaiを使う方法
Google Colabでは標準でfastaiが入っているため、簡単にfastaiを使うことができます。
```
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```
2行目以降にエラーが出るようであれば、GPUが使われているか確認し、ランタイムを再起動して見ていください。自分はそれでエラーがなくなりました。  
また、自身のGoogle Driveに接続するという旨のメッセージが出るので接続してください。  
これでfastaiが使えるようになったので、以下のようにインポートすることができるようになります。
```
from fastai.vision.all import *
```
### CPUのローカルPCでfastaiを使う方法
正直、ここでかなり疲れました。いろいろやって、最終的にローカルで作った人の記事を参考にしたらできました(最初からそれをやればよかった)。下の方に転載しておきますね。  
では、一つずつ順番に見ていきましょう。ちなみにVSCodeでやりました。  
まず最初にVSCodeのターミナルで以下のようにgitでクローンします。
```
git clone https://github.com/fastai/fastai.git
```
こうすることで、fastaiのフォルダが追加されました。
そしたら、今いるディレクトリを以下のようにfastaiに移動し、仮想環境を変えます。  
仮想環境は作らずに、git cloneして自動で作られるので安心してください。
```
cd fastai
activate fastai (conda activate fastai)
```
次は、順番に注意して、PyTorchのcpu版をインストールします。環境によって、コマンドが変わると思いますので気を付けてください。詳細は[ここから](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
PyTorchを入れ終わったら、fastaiフォルダの中にenvironment.ymlが入っているので、
```
- pytorch
```
という記述を削除します。削除したら、
```
conda env update -f environment.yml
```
と入力し実行します。 
やっと終わりが見えてきました。  
次に大事なことは、VSCodeの右上のインタプリタをfastaiに変更してください。  
![image](https://user-images.githubusercontent.com/106716245/183022327-d7fae55a-835f-4c09-a302-8515e6400c7d.png)  
次で、インストールするのが終わりです。
```
pip install -Uqq flaskbook
```
と入力してください。何も出力されなければ、大丈夫です。  
続いて、fastaiのフォルダ内でipynbファイルを作成してください。ここで、
```
import flaskbook
flaskbook.setup_book()
from flaskbook import *
```
と無事実行できれば終了です。お疲れさまでした。

ここまでできれば、[fastaiのドキュメント](https://docs.fast.ai/)に書かれたコードを実行することができると思います。  インストールの仕方が違うので、もしこのやり方で無理でしたら、公式の方でやってもらった方がよいと思います。
いやー長かった(笑)。  
実行はできたけれど、CPUだからめちゃくちゃ遅い(笑)。  
ぜひとも、参考になればうれしいです。

### 参考文献
https://qiita.com/ishida330/items/34b28fc18f66d98a0479  
https://docs.fast.ai/  
https://forums.fast.ai/t/howto-installation-on-windows/10439
