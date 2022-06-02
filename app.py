from flask import Flask, render_template, request, Markup,jsonify
import numpy as np
import pandas as pd
import pickle
import requests
import io

crop_recommendation_model_path = 'C:/Users/hp/Desktop/Final_App/models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)

@ app.route('/crop_recommend', methods=['POST'])
def crop_recommend():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return jsonify({'label':str(final_prediction)})
        else:
            return jsonify({'label':'Data not available'})
