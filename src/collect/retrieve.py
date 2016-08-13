import json
import urllib

import math

import constants as const


####################################################################
# Bjin.MeのAPIから画像を取得・保存する（ランダム）
#  返り値:
#    data: 取得した画像を辞書型で返却する
#        ID：コンテンツ特定ID
#        category：人名（空の場合あるっぽい）
#        thumb：画像（サムネイル）
#        link：表示Link
#        pubData：データが発行された日時
#        imgName：画像名（XXXXX.jpg）
####################################################################
def retrieve():
    print('Start retrieve')
    # 画像を取得
    url = 'http://bjin.me/api/?type=rand&count=' + str(const.IMAGES_NUM) + '&format=json'
    html = urllib.request.urlopen(url).read()
    data = json.loads(html.decode('utf8'))

    # JSONで取得した画像リストを1件ずつ処理
    for photo in data:
        thumb = photo['thumb']
        # サムネイルの画像名部分（XXXXX.jpg）を取得
        imgName = thumb.split('/')[-1]
        # 辞書型に画像名を表すimgName要素を追加
        photo['imgName'] = imgName
        # 実験用に画像を2箇所に保存
        urllib.request.urlretrieve(thumb, const.PATH_RET + imgName)
        urllib.request.urlretrieve(thumb, const.PATH_RECT + imgName)

    print('End retrieve')
    return data

####################################################################
# Bjin.MeのAPIからトレーニング画像を取得する
# トレーニング済みの場合は画面出力しない
#  返り値:
#    data: 取得した画像を辞書型で返却する
#        ID：コンテンツ特定ID
#        category：人名（空の場合あるっぽい）
#        thumb：画像（サムネイル）
#        link：表示Link
#        pubData：データが発行された日時
#        imgName：画像名（XXXXX.jpg）
####################################################################
def getTrainingImg():
    url = 'http://bjin.me/api/?type=rand&count=1&format=json'

    print('Start getTrainingImg')
    while True:
        # 画像を取得
        html = urllib.request.urlopen(url).read()
        data = json.loads(html.decode('utf8'))

        photo = data[0]
        print(photo)
        # 画像が仕分け済みの場合、画面表示しない
        if isAlreadyTraining(str(math.floor(photo['id']))):
            continue
        thumb = photo['thumb']
        # サムネイルの画像名部分（XXXXX.jpg）を取得
        imgName = thumb.split('/')[-1]
        # 辞書型に画像名を表すimgName要素を追加
        photo['imgName'] = imgName
        # 実験用に画像を2箇所に保存
        urllib.request.urlretrieve(thumb, const.PATH_RET + imgName)
        urllib.request.urlretrieve(thumb, const.PATH_RECT + imgName)
        break
    print('End getTrainingImg')

    return data

####################################################################
# 画像番号が仕分け済みか判定する
#  引数:
#    num: 画像番号（string）
#  返り値:
#    判定結果: True（仕分け済み）、False（未仕分け）
####################################################################
def isAlreadyTraining(num):
    print('isAlreadyTraining: ' + num)

    f = open(const.PATH_TRIM + 'train.txt', 'r')
    for line in f:
        line = line.rstrip()
        l = line.split()
        # train.txtに記載されていない画像の場合、
        if num == l[0].replace('.jpg', ''):
            return True
    return False
