from flask import Flask,render_template,request,url_for
import pandas as pd
from pycaret.regression import *
import numpy as np
import pickle

app=Flask(__name__)
model=load_model('deployment')
cols=['age','sex','bmi','children','smoker','region']


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    dat_useen=pd.DataFrame([final_features],columns=cols)
    prediction = predict_model(model,data=dat_useen,round=0)
    prediction = int(prediction.Label[0]*370)
    return render_template("home.html", prediction_text='charge should be # {}'.format(prediction)+'NGN')



if __name__ == '__main__':
    app.run(debug=True)
