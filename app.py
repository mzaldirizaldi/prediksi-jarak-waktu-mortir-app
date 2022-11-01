from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# fitting scaler using pandas
filepath = "tabel_tembak.csv"
data_tembak = pd.read_csv(filepath)
data_tembak_processed = data_tembak.drop(['tipe', 'jarak_m', 'waktu_lintas_det'], axis=1)
scaler = StandardScaler()
scaler.fit(data_tembak_processed)

# load models
model_waktu = load('model_waktu.pkl')
model_jarak = load('model_jarak.pkl')
scaler_jarak = StandardScaler()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    # getting data from post
    isian = request.form.get('isian')
    isian = int(isian)

    derajat = request.form.get('sudut_elevasi_der')
    derajat = float(derajat)
    radian = 0
    input_query = (isian, derajat, radian)

    # convert derajat to rad
    input_list = list(input_query)
    pi = 22 / 7
    radian = input_list[1] * (pi / 180)
    radian = round(radian * 1000)
    input_list[2] = radian
    input_query = tuple(input_list)

    # convert to array and reshape
    input_data_as_numpy = np.asarray(input_query)
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # standardize data
    std_data = scaler.transform(input_data_reshaped)

    # predict
    result_waktu = model_waktu.predict(std_data)
    result_jarak = model_jarak.predict(std_data)
    result_waktu = float(result_waktu)
    result_jarak = int(result_jarak)
    pred_2f_waktu = "{:.2f}".format(result_waktu)

    #send as json
    return jsonify({'jarak_m':str(result_jarak),'waktu_lintas_det':str(pred_2f_waktu)})

@app.route('/predicthtml',methods=['POST'])

def predicthtml():
    # getting data from post
    isian = request.form.get('isian')
    isian = int(isian)

    derajat = request.form.get('sudut_elevasi_der')
    derajat = float(derajat)
    radian = 0

    input_query = (isian, derajat, radian)

    # convert derajat to rad
    input_list = list(input_query)
    pi = 22 / 7
    radian = input_list[1] * (pi / 180)
    radian = round(radian * 1000)
    input_list[2] = radian
    input_query = tuple(input_list)

    # convert to array and reshape
    input_data_as_numpy = np.asarray(input_query)
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # standardize data
    std_data = scaler.transform(input_data_reshaped)

    # predict
    result_waktu = model_waktu.predict(std_data)
    result_jarak = model_jarak.predict(std_data)
    result_waktu = float(result_waktu)
    result_jarak = int(result_jarak)
    pred_2f_waktu = "{:.2f}".format(result_waktu)

    #send as html
    return render_template('index.html', jarak_m='Jarak: {} meter'.format(result_jarak), waktu_lintas_det='Waktu Lintas: {} detik'.format(pred_2f_waktu))

if __name__ == '__main__':
    app.run(debug=True)
