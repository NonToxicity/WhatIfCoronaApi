import flask
from flask import request
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy


loaded_model = None
label_encoder = None
oneHotEncoder = None
app = flask.Flask(__name__)

@app.route('/',methods=['GET'])


def home():
    
    arg = request.args['predict']
    load()
    if(loaded_model != None):
        result = arg
        #result = loaded_model.predict([[1,1,1]])
    return str(result)


def load():
    global loaded_model
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classesLabelEncoder.npy')
    oneHotEncoder = joblib.load('encoder.joblib')
        
