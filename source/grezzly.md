---
title: データを集めて、画像分類しよう
date: 2022-08-05
tags: ["画像分類", "fastai"]
excerpt: azureから画像データを集めて、画像分類をやってみよう
---

# 初めての画像分類
この記事では「PyTorchとfastaiによるディープラーニング」をもとにコードとその解説を記述しています。    
以下のコードはGoogle Colabを想定しています。皆さんもコードをいろいろいじってみてください。

## ライブラリをインポートしよう
---
まず初めに、fastaiをインポートします。
```python3
! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```
``` python3
from fastbook import *
from fastai.vision.all import *
```

## 画像を集めよう
---
今回はazureのBing Image Searchから画像を集めていきたいと思います。  
150画像を一気にダウンロードできるので十分だと思います。  
xxxには自身のキーを入力してください。
```python3
key = os.environ.get('AZURE_SEARCH_KEY', 'xxx')
```
キーをセットすることで、 search_images_bing関数が使えるようになります。 
search_images_bing関数の引数は、azureのキーと調べ対単語を入力します。イメージとしてはGoogleで *grizzly bear* と入力するということです。 
では、実際にsearch_images_bing関数を使ってみましょう。
```python3
results = search_images_bing(key, 'grezzly bear') 
ims = results.attrgot('contentUrl')
len(ims)

>>> 150
```
次に、グリズリーの画像を1枚ダウンロードしてみましょう。
```python3
dest = 'images/grizzly.jpg'
download_url(ims[0], dest)

im = Image.open(dest)
im.to_thumb(128, 128) # サイズを128x128に変更
```

ではここから、グリズリーとクロクマ、テディベアの画像を集めていきます。
```python3
bear_types = 'grizzly','black','teddy'
path = Path('bears')
```
```python3
if not path.exists():　# それぞれ単語に対するフォルダが存在するかどうか判断
    path.mkdir() 
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear') # 単語に対するURLをダウンロードする
        download_images(dest, urls=results.attrgot('contentUrl'))

fns = get_image_files(path)
```
そして、ネットからの画像なのでファイルが破損している場合があるので、チェックします。
```python3
failed = verify_images(fns) # 破損しているファイルを取得
failed.map(Path.unlink);    # 取得した画像を破棄
```
次に、取得した画像を訓練用とテスト用に分けます。
```python3
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)
```
上記のコードについて1つずつ説明します。
>&nbsp;&nbsp;blocks = (独立変数, 従属変数)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;独立変数: 予測に用いる変数, 画像のセット  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;従属変数: 予測のターゲットとなる変数, 個々の画像のカテゴリ(熊の種類)

> &nbsp;&nbsp;get_items = get_image_files  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ファイルがあるパスを受け取り、そのパスの下にある全ての画像を取得してリストとして返す

>&nbsp;&nbsp;splitter=RandomSplitter(valid_pct=0.2, seed=42)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;訓練セットと検証セットをランダムに分割する  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;valid_pct : 検証セットの割合
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seed : 乱数のシードを設定, 毎回同じ数値が得られる

> &nbsp;&nbsp;get_y=parent_label  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;そのファイルが格納されているフォルダ名をそのままラベルとして使用する

>&nbsp;&nbsp;item_tfms=Resize(128)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;アイテム変換を行い、画像を128pxにリサイズする

次にデータの水増しを行っていきます。aug_transforms関数を使用し、自動で回転や反転、コンストラクトの変更などを行ってくれる。
```python3
bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)
```

##訓練
データ収集から加工をやってきました。長いね。
やっと、モデルの訓練をすることができます。
```python3
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```
これを実行すると、4エポックの結果が出力されると思います。  
次に、どのように間違っているのか可視化してみましょう。
```python3
from fastai.vision import learner
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```
横軸が予測、縦軸が正解を表します。  
最後に画像のロスの高い順に並べてみましょう。
```python3
interp.plot_top_losses(5, nrows=1)
```
左から予測した値、実際のラベル、ロス、確率が出ています。　　

### 最後に
お疲れさまでした。
長々と書いてきましたが、特に難しいことをしているわけではないことに気づいたではないでしょうか。  
このぐらいの画像分類だったら、自分でいろいろとできると思うのでやってみるといいと思います。  
