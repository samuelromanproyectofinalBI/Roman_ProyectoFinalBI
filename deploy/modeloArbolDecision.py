import time
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st

def app():
    plt.style.use('fivethirtyeight')

    header = st.header('Cabecera')

    with header:
        st.title('Cabecera')
        st.write('Conjunto de datos')

    # Definimos el tamaño de la figura
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 20,10

    # Normalizamos los datos
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    ticker = 'DIS'
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)

    df['symbol'] = 'DIS'
    df.index = pd.to_datetime(df['Date'])
    df = df.drop(['Date'], axis='columns')
    st.subheader('Datos del conjunto de datos')
    st.write(df)

    df = df[['Close']]
    st.subheader('Columna "Close"')
    st.write(df)

    # Creamos una variable para predecir los precios 'x' días en el futuro
    future_days = 100
    df['Prediction'] = df[['Close']].shift(-future_days)
    st.subheader('Columna "Prediction"')
    st.write(df)

    X = np.array(df.drop(['Prediction'], 1))[:-future_days]
    st.subheader('Variable independiente X')
    st.write(X)

    y = np.array(df['Prediction'])[:-future_days]
    st.subheader('Variable dependiente y')
    st.write(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    tree = DecisionTreeRegressor().fit(x_train, y_train)

    x_future = df.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    st.subheader('Variable X para la predicción futura')
    st.write(x_future)

    tree_prediction = tree.predict(x_future)
    st.subheader('Predicción utilizando el modelo Decision Tree')
    st.write(tree_prediction)

    rms = np.sqrt(np.mean(np.power((x_future - tree_prediction), 2)))
    st.subheader('Error cuadrático medio (RMSE)')
    st.write(rms)

    # Visualizamos los datos
    st.subheader('Visualización de los datos')
    predictions = tree_prediction

    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig = plt.figure(figsize=(16, 8))
    plt.title('Modelo Decision Tree')
    plt.xlabel('Días')
    plt.ylabel('Precio de cierre en USD($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Datos originales', 'Datos válidos', 'Datos predichos'])

    st.pyplot(fig)