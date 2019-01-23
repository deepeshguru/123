# A very simple Flask Hello World app for you to get started with...

from flask import Flask, jsonify, request
import bz2
import pickle
import pandas as pd

with bz2.BZ2File('sentiment_cv.pbz2', 'r') as f:
    cv = pickle.load(f)

with bz2.BZ2File('sentiment_model.pbz2', 'r') as f:
    model = pickle.load(f)


app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>Hello Deepesh Sentiment Analysis</h1>"


@app.route("/analysis/", methods=['POST'])
def analysis():

    x = request.get_json()
    com = {
        'comments': x['comment']
        }
    tt = []
    for i in range(len(com['comments'])):
        tt.append(str(com['comments'][i]))

    tt = pd.Series(tt)

    xcv = cv.transform(pd.Series(tt))
    pred = model.predict(xcv)
    k = []
    for i in range(len(tt)):
        k.append({'comment': str(tt[i]), 'sentiment': str(pred[i])})
    jj = {'text_analysis': k}
    return jsonify(jj)


