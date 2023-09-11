from flask_restful import Resource, Api, reqparse, abort
from flask import Flask, request
import numpy as np
import pandas as pd
import pickle as pk
import joblib as jb
import tensorflow as tf
from tensorflow import keras
import json

app = Flask(__name__)
api = Api(app)
model = keras.models.load_model('./Resources/lstm_model.keras')
pca = pk.load(open('./Resources/pca.pkl','rb'))
drop_columns = pk.load(open('./Resources/drop_columns.pkl', 'rb'))
drop_columns.remove('Unnamed: 0')
scaler = pk.load(open('./Resources/scaler.pkl','rb'))


class MLAlgorithm(Resource):

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(),list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names +=[('pca%d(t-%d)' %(j+1, i)) for j in range (n_vars)]
        #forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names +=[('pca%d(t)' %(j+1)) for j in range (n_vars)]
            else:
                names +=[('pca%d(t+%d)' %(j+1, i)) for j in range (n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns=names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg




    def post(self):
        X = request.get_json()
        df = pd.DataFrame(X)
        df = df.drop(['Id','timestamp'], axis =1)
        df = df.drop(labels=drop_columns, axis =1)
        df.fillna(method='ffill', inplace=True)
        scaled_df = scaler.transform(df)
        pComponents = pca.transform(scaled_df)
        principal_df = pd.DataFrame(data = pComponents,columns=['pca1','pca2', 'pca3','pca4','pca5', 'pca6','pca7','pca8', 'pca9','pca10','pca11', 'pca12','pca13','pca14', 'pca15',])
        Lag=9 # How many steps to look into the future
        data_shift= self.series_to_supervised(data=principal_df, n_in=Lag, n_out=1)
        lstm_x = data_shift.values.reshape((data_shift.shape[0], 1, data_shift.shape[1]))
        prediction = model.predict(lstm_x)
        y_pred = self.assign_classes(prediction)
        return  float(y_pred[0, 0])

    def assign_classes(self, y_pred_):
        y_pred = y_pred_.copy()
        y_pred[y_pred<1/2] = 0 # BROKEN
        y_pred[y_pred>=1/2] = 1 # NORMAL
        return y_pred

    def __init__(self) -> None:
        super().__init__()

api.add_resource(MLAlgorithm, "/MLAlgorithm")

if __name__ =="__main__":
    app.run(debug=True)
