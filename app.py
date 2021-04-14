import flask
from flask import request
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy
from sklearn import *


loaded_model = None
label_encoder = None
oneHotEncoder = None
app = flask.Flask(__name__)

@app.route('/',methods=['GET'])


def home():
    global loaded_model
    global label_encoder
    global oneHotEncoder
    arg = request.args['predict']
    load()
    if(loaded_model != None):

        prediction = arg
        predicted = prediction.split(",")
        #predicted = ['Afghanistan',300,37.746,2.581,8.33,38928341,0,0,0]
        a = numpy.asarray(predicted)

        df = pd.DataFrame([a], columns = ['location','date','handwashing_facilities','aged_65_older','stringency_index','population','total_vaccinations','female_smokers','male_smokers'])

        categorical_cols = ['location'] 

        # apply le on categorical feature columns
        df[categorical_cols] = df[categorical_cols].apply(lambda col: label_encoder.transform(col))    

        # apply one hot encoding to categorical
        array_hot_encoded = oneHotEncoder.transform(df[categorical_cols])
        data_hot_encoded = pd.DataFrame(array_hot_encoded)
        
        #Extract only the columns that didnt need to be encoded
        data_other_cols = df.drop(columns=categorical_cols)

        #Concatenate the two dataframes : 
        data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)

        df = data_out
        df = df.fillna(0) 
        df = df.applymap(str)
        df = df.applymap(float)

        temp = df.iloc[0]
        temp = temp.values.reshape(1,-1) 
        
        result = loaded_model.predict(temp)

    return str(result)


def load():
    global loaded_model
    global label_encoder
    if(label_encoder == None):
        global oneHotEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = numpy.load('classesLabelEncoder.npy',allow_pickle=True)
        oneHotEncoder = joblib.load('encoder.joblib')
    if(loaded_model == None):
        loaded_model = pickle.load(open('finalized_model.sav', 'rb'))