import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.svm import SVC
from sklearn.metrics import classification_report

def app():

    st.title('Modelo SVC')
    ticker = 'META'
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_dis = pd.read_csv(query_string)

    st.subheader("Filtramos por símbolo META y obtenemos el dataframe")
    df_dis['symbol'] = 'META'
    st.write(df_dis)

    st.subheader("Creación de variables de predicción")
    df_dis['Open-Close'] = df_dis.Open - df_dis.Close
    df_dis['High-Low'] = df_dis.High - df_dis.Low

    st.subheader("Guardamos los valores relevantes en la variable X")
    X = df_dis[['Open-Close', 'High-Low']]
    st.write(X.tail(4))

    st.subheader("Haciendo la definición del objetivo {0} o {1}")
    y = np.where(df_dis['Close'].shift(-1) > df_dis['Close'], 1, 0)
    st.write(y)

    st.subheader("Dividir los datos en entrenamiento y prueba")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    st.write("x_train:", x_train)
    st.write("x_test:", x_test)
    st.write("y_train:", y_train)
    st.write("y_test:", y_test)

    st.subheader("Entrenamiento del modelo")
    modelo = SVC().fit(x_train, y_train)
    st.write(modelo)

    st.subheader("Haciendo predicción según datos de prueba")
    y_predict = modelo.predict(x_test)
    st.write(y_predict)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_predict, output_dict=True)
    df_report = pd.DataFrame(report)
    st.write(df_report)

    st.subheader("Realizamos una prueba con datos ingresados")
    test = [[1.160004 , 2.430001],[-0.110001, 1.050004]]
    df = pd.DataFrame(test, columns=['Open-Close', 'High-Low'])
    y_predict = modelo.predict(df)
    st.write(y_predict)