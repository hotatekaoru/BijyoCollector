import cv2
import numpy as np
import glob
import constants as const

####################################################################
# OpenCVを使用して顔の部分を切出す
#  引数:
#    dicts: 取得した画像情報
####################################################################
def trim(dicts):
    print("Start trim")
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    white = (255, 255, 255)
    count = 0

    # データを入れる配列
    for dict in dicts:
        # ファイル読み込み
        filePath = const.PATH_TRIM + dict["imgName"]
        image = cv2.imread(filePath)
        # グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), white, thickness=2)

            # 認識結果の保存
            cv2.imwrite(filePath, image)
            count += 1

    print("画像取得数: " + str(len(dicts)))
    print("顔認識成功数: " + str(count))
    # データを読み込んで28x28に縮小
        #img = cv2.imread(file)
        #img = cv2.resize(img, (28, 28))
        # 一列にした後、0-1のfloat値にする
        #train_image.append(img.flatten().astype(np.float32) / 255.0)
        # ラベルを1-of-k方式で用意する
        #tmp = np.zeros(2)
        #tmp[int(l[1])] = 1
        #train_label.append(tmp)
        #count += 1
        #file.close()
    # numpy形式に変換
    #train_image = np.asarray(train_image)
    #train_label = np.asarray(train_label)
    #print(train_image)
    #print(train_label)

    return
