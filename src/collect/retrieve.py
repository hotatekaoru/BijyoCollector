import json
import urllib
import constants as const
import os.path

####################################################################
# Bjin.MeのAPIから画像を取得・保存する
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
    print("Start retrieve")
    # 画像を取得
    url = 'http://bjin.me/api/?type=rand&count=' + str(const.IMAGES_NUM) + '&format=json'
    train_url = 'http://bjin.me/api/?type=detail&count=1&format=json&id=' + str(const.IMAGES_NUM)
    html = urllib.request.urlopen(url).read()
    data = json.loads(html.decode('utf8'))

    # JSONで取得した画像リストを1件ずつ処理
    for photo in data:
        thumb = photo["thumb"]
        # サムネイルの画像名部分（XXXXX.jpg）を取得
        imgName = thumb.split('/')[-1]
        # 辞書型に画像名を表すimgName要素を追加
        photo["imgName"] = imgName
        # 実験用に画像を2箇所に保存
        urllib.request.urlretrieve(thumb, const.PATH_RET + imgName)
        urllib.request.urlretrieve(thumb, const.PATH_RECT + imgName)

    print(data)
    print("End retrieve")
    return data

def searchImgId():
    print("Start searchImgId")
    count = 1
    while os.path.isfile(const.PATH_RET + str(count) + '.jpg'):
        count += 1

    print("End searchImgId")
    return str(count)
