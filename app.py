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
        predicted = ['Afghanistan',0,37.746,2.581,8.33,38928341,0,0,0]
        a = numpy.asarray(predicted)

        df = pd.DataFrame(my_array, columns = ['location','date','handwashing_facilities','aged_65_older','stringency_index','population','total_vaccinations','female_smokers','male_smokers'])

        categorical_cols = ['location'] 
        # apply le on categorical feature columns
        df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))    

        array_hot_encoded = ohe.fit_transform(df[categorical_cols])
        
        #Convert it to df
        data_hot_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        #Extract only the columns that didnt need to be encoded
        data_other_cols = df.drop(columns=categorical_cols)
        
        #Concatenate the two dataframes : 
        data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)

        data_out = data_out.applymap(float)
        
        result = loaded_model.predict(data_out)
    return str(result)


def load():
    global loaded_model
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    label_encoder = LabelEncoder()
    label_encoder.classes_ = numpy.load('classesLabelEncoder.npy',allow_pickle=True)
    oneHotEncoder = joblib.load('encoder.joblib')
        
