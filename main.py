import src.collect.retrieve as rt
import src.collect.trim as tr
import time

def retrieve():
    return rt.retrieve()
def trim(ret_images):
    return tr.trim(ret_images)

# webapp
from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.route('/api/collect', methods=['POST'])
def collect():
    start_time = time.time()
    print("開始時刻: " + str(start_time))

    # 画像取得
    ret_images = retrieve()
    # 顔認識
    trim(ret_images)

    end_time = time.time()
    print("終了時刻: " + str(end_time))
    print("かかった時間: " + str(end_time - start_time))
    return jsonify(results=ret_images)


@app.route('/')
def main():
    return render_template('index.html')
