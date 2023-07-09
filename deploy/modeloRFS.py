import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def app():
    st.title('Modelo Random Forest Regression')

    # Obtener datos de Yahoo Finance
    st.subheader("Obtener datos de Yahoo Finance")
    ticker = 'DIS'
    period1 = int(time.mktime(datetime.datetime(2015, 1, 1, 0, 0).timetuple()))
    period2 = int(time.mktime(datetime.datetime.now().timetuple()))
    interval = '1d' # 1d, 1m
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_dis = pd.read_csv(query_string)

    df_dis['symbol'] = 'DIS'
    st.write(df_dis)

    st.subheader("Preprocesamiento de datos")

    # Establecer el índice como fecha
    st.write('Establecemos el índice como fecha y graficamos')
    df_dis['Date'] = pd.to_datetime(df_dis.Date, format='%Y-%m-%d')
    df_dis.index = df_dis['Date']

    # Realizar plot
    fig = plt.figure(figsize=(16, 8))
    plt.plot(df_dis['Close'], label='Precio de Cierre Histórico')
    st.write(fig)

    # Guardar en un DataFrame los datos de la columna Close
    st.write('Guardar en un DataFrame los datos de la columna Close')
    df = df_dis[['Close']]
    st.write(df.tail(4))

    st.subheader("Definir la variable predictora")
    st.write('Mostramos el conjunto de datos con la columna Prediction')
    # Crear una variable para predecir 'x' días en el futuro
    future_days = 100
    # Crear una nueva columna (objetivo) desplazada 'x' unidades/días hacia arriba
    df['Prediction'] = df[['Close']].shift(-future_days)
    st.write(df)

    st.write("Creamos el conjunto de datos de características (X) y lo convertimos en un numpy array y eliminamos las últimas 'x' filas/días")
    X = np.array(df.drop(['Prediction'], 1))[:-future_days]
    st.write(X)

    st.write("Creamos el conjunto de datos objetivo (y) en un numpy array y obtenemos todos los valores target excepto las últimas filas")
    y = np.array(df['Prediction'])[:-future_days]
    st.write(y)

    st.write("Dividimos los datos en 75% para entrenamiento y 25% para prueba")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    st.subheader("Entrenar el modelo")

    st.write("Importamos el Random Forest Regressor")
    from sklearn.ensemble import RandomForestRegressor

    st.write("Creamos un objeto de tipo Random Forest Regressor")
    RFReg = RandomForestRegressor(n_estimators=100, random_state=0)

    st.write("Ajustamos el Random Forest Regressor con los datos de entrenamiento representados por x_train y y_train")
    RFReg.fit(x_train, y_train)

    # Obtener las últimas filas 'x' del conjunto de datos futuros
    x_future = df.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    # Altura prevista del conjunto de datos de prueba
    random_forest_regressor = RFReg.predict(x_future)
    st.write(random_forest_regressor)

    st.subheader("Visualizar los datos")
    predictions = random_forest_regressor

    valid = df[X.shape[0]:]
    valid['Predictions'] = predictions
    fig = plt.figure(figsize=(16, 8))
    plt.title('Random Forest Regressor')
    plt.xlabel('Días')
    plt.ylabel('Precio de Cierre en USD($)')
    plt.plot(df['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Datos Originales', 'Datos Válidos', 'Datos Predichos'])
    plt.show()
    st.write(fig)