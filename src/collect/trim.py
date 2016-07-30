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
    success = 0

    # データを入れる配列
    for dict in dicts:
        # ファイル読み込み
        rectPath = const.PATH_RECT + dict["imgName"]
        trimPath = const.PATH_TRIM + dict["imgName"]
        image = cv2.imread(rectPath)

        # グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        print(facerect)

        if len(facerect) > 0:
            # 検出した顔を囲む矩形の作成
            for rect in facerect:
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), white, thickness=2)

            # 画像の保存
            cv2.imwrite(rectPath, image)

            # 画像のトリミング
            for rect in facerect:
                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

                trimImage = image[y:y + h, x:x + w]

                # 画像の保存
                cv2.imwrite(trimPath, cv2.resize(trimImage, (const.CLASSIFY_IMG_SIZE_PX, const.CLASSIFY_IMG_SIZE_PX)))

            success += 1

    print("画像取得数: " + str(len(dicts)))
    print("顔認識成功数: " + str(success))

    return