import src.collect.retrieve as rt
import src.collect.trim as tr
import time

def retrieve():
    return rt.retrieve()
def getTrainingImg():
    return rt.getTrainingImg()
def searchImgId():
    return rt.searchImgId()
def isAlreadyTraining(num):
    return rt.isAlreadyTraining(num)
def trim(ret_images):
    return tr.trim(ret_images)

# webapp
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/api/collect', methods=['POST'])
def collect():
    start_time = time.time()
    print('開始時刻: ' + str(start_time))

    # 画像取得
    ret_images = retrieve()
    # 顔認識
    trim(ret_images)

    end_time = time.time()
    print('終了時刻: ' + str(end_time))
    print('かかった時間: ' + str(end_time - start_time))
    return jsonify(results=ret_images)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/training')
def transitTraining():
    return render_template('training.html')

@app.route('/api/training', methods=['POST'])
def training():
    return jsonify(results=getTrainingImg())