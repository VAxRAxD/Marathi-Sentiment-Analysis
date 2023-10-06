import joblib, json, base64, cv2, numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
import utils

__model = None

def analyzeSentiment(text):
  data = [utils.Preprocessing_for_Marathi_Language(text)]
  inp=utils.convert(data)
  ans=__model.predict(inp)
  return ans[0]

def load_saved_artifacts():
    print("loading saved artifacts...start")

    global __model
    if __model is None:
                with open('./artifacts/sent_analysis.pkl', 'rb') as f:
                    __model = joblib.load(f)
    print("loading saved artifacts...done")
 
app = Flask(__name__)   

@app.route('/', methods=['GET', 'POST'])
def logistic_regression():
    if request.method == "POST":
        txt=request.form.get("data")
        sent=analyzeSentiment(txt)
        print(sent)
        return {"sentiment": str(sent)}
    return render_template("index.html")

if __name__ == '__main__':
    load_saved_artifacts()
    app.run(port=5000,debug=True)