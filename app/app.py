import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA
import pandas as pd
import json
import numpy as np
import math
from joblib import load

def calculateEntropy(x):
    counter = {}
    for i in x:
        counter[i]  = counter.get(i, 0) + 1
    entropy = 0
    log = math.log2
    n = len(x)
    for key in counter:
        p = counter[key]/n
        entropy = entropy - (p * log(p))
    return entropy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    try:
        uploaded_file = request.files['file']
        print(type(uploaded_file))
    except:
        return redirect(url_for('index'))
    if not uploaded_file:
        return redirect(url_for('index'))
    
    data, data_dict = show_tables(uploaded_file)
    recommendations = getRecommendation(data_dict)
    return render_template('view.html',table=data.to_html(classes='meta'), title = 'Meta Information', recommendations = recommendations)
    # return redirect(url_for('index'))

def processInfo(df):
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    res = {}
    res["class_int"] = len(set(y))/(x.shape[0])
    res["attr_class"] = (x.shape[1])/len(set(y))
    pca = PCA(4)
    x = pca.fit_transform(x)
    df_x = pd.DataFrame(data = x)
    skew = df_x.skew()
    for i in range(4):
        res["skewnes_" + str(i)] = skew[i]
    description = df_x.describe()
    for i in range(4):
        res["mean_"+str(i)] = description.loc['mean', i]
    for i in range(4):
        res["std_"+str(i)] = description.loc['std', i]
    for i in range(4):
        res["median_"+str(i)] = np.median(x[:, i])
    y = y.values
    res["class_entropy"] = calculateEntropy(y)
    return res

def show_tables(uploaded_file):
    df = pd.read_csv(uploaded_file)
    data = processInfo(df)
    attr = []
    val = []
    for key in data:
        attr.append(key)
        val.append(data[key])
    
    dataDF = pd.DataFrame()
    dataDF['Attribute'] = attr
    dataDF['Value'] = val
    dataDF.set_index(['Attribute'], inplace=True)
    dataDF.columns.name = dataDF.index.name
    dataDF.index.name = None
    return dataDF,data

def getRecommendation(data_dict):
    assert(type(data_dict) == dict)
    columns = app.config['COLUMNS']
    model = app.config['ML_MODEL']
    classes = columns[-4:]
    for algo in classes:
        data_dict[algo] = 0
    data = [data_dict.copy() for i in range(len(classes))]
    for i in range(len(classes)):
        data[i][classes[i]] = 1
    df = pd.DataFrame(data)
    df = df[columns]
    values = df.values
    pred = model.predict(values)
    data_pair = [(pred[i], classes[i]) for i in range(len(classes))]
    data_pair = sorted(data_pair, key = lambda x : x[0], reverse=True)
    result = [x[1].split("_")[-1] for x in data_pair]
    return result

app.config['UPLOAD_EXTENSIONS'] = ['.csv']
app.config['MODEL_FILE_PATH'] = "assets/model.joblib"
app.config['ML_MODEL'] = load(app.config['MODEL_FILE_PATH'])
app.config['COLUMNS_FILE_PATH'] = "assets/columns.txt"
with open(app.config['COLUMNS_FILE_PATH'], 'r') as f:
    app.config['COLUMNS']  = json.load(f)[:-1]
    if __name__ == "__main__":
        app.run(debug=True)