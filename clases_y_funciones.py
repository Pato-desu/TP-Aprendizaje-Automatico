import pandas as pd
import numpy as np
# import math
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import xticks
# import matplotlib.dates as mdates

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_absolute_error
# from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
# from sklearn.metrics import roc_curve, roc_auc_score, auc
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import label_binarize
# from sklearn.multiclass import OneVsRestClassifier
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# import shap
# import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.metrics import F1Score
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.layers import Flatten, Dense
# from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

dir2ang = {
    'E': 0,
    'ESE': 22.5,
    'SE': 45,
    'SSE': 67.5,
    'S': 90,
    'SSW': 112.5,
    'SW': 135,
    'WSW': 157.5,
    'W': 180,
    'WNW': 202.5,
    'NW': 225,
    'NNW': 247.5,
    'N': 270,
    'NNE': 292.5,
    'NE': 315,
    'ENE': 337.5
}

dict_epochs_types={
    "_250_epocas_largas":{
        "epochs": 250,
        "batch_size": 100},    #batchs: x_train: 122, x_train2: 98
    "_500_epocas_medianas":{
        "epochs": 500,
        "batch_size": 510},    #batchs: x_train: 24, x_train2: 20
    "_1000_epocas_cortas":{
        "epochs": 800,
        "batch_size": 2500}    #batchs: x_train: 5, x_train2: 4
}

# Custom transformer for imputation that returns a DataFrame
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        transformed_array = self.imputer.transform(X)
        return pd.DataFrame(transformed_array, columns=X.columns)
    
class StandardScalerFor3Tuple:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X, y1=None, y2=None):
        if y1 is None:
            y1 = X[1]
            y2 = X[2]
            X = X[0]

        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        scaled_X = (X - self.mean_) / self.scale_
        return scaled_X.values, y1.values, y2.values

    def fit_transform(self, X, y1=None, y2=None):
        if y1 is None:
            y1 = X[1]
            y2 = X[2]
            X = X[0]

        return self.fit(X).transform(X, y1, y2)
    
class ClassificationNeuralNetwork:
    def __init__(self, n_hidden_layers, n_neurons, activations, learning_rate, epochs_type=None, epochs=None, batch_size=None, metric=None):
        # Guardo los datos de los epochs para el training
        if n_hidden_layers > 2:
            print("Demasiadas capas ocultas")
            return None
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        if epochs_type is not None:
            self.epochs = dict_epochs_types[epochs_type]["epochs"]
            self.batch_size = dict_epochs_types[epochs_type]["batch_size"]

        # Inicializo el modelo
        self.model = Sequential()
        match n_hidden_layers:
            case 0:
                self.model.add(Dense(1, activation='sigmoid', input_shape=(m,)))
            case 1:
                self.model.add(Dense(n_neurons[0], activation=activations[0], input_shape=(m,)))
                self.model.add(Dense(1, activation='sigmoid'))
            case 2:
                self.model.add(Dense(n_neurons[0], activation=activations[0], input_shape=(m,)))
                self.model.add(Dense(n_neurons[1], activation=activations[1]))
                self.model.add(Dense(1, activation='sigmoid'))

        # Compilar el modelo
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=[metric])

    def fit(self, x_train, y_train, epochs=None, batch_size=None, validation_data=None):
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
    
class RegressionNeuralNetwork:
    def __init__(self, n_hidden_layers, n_neurons, activations, learning_rate, epochs_type=None, epochs=None, batch_size=None, metric=None):
        # Guardo los datos de los epochs para el training
        if n_hidden_layers > 2:
            print("Demasiadas capas ocultas")
            return None

        self.epochs = epochs
        self.batch_size = batch_size
        if epochs_type is not None:
            self.epochs = dict_epochs_types[epochs_type]["epochs"]
            self.batch_size = dict_epochs_types[epochs_type]["batch_size"]

        # Inicializo el modelo
        self.model = Sequential()
        match n_hidden_layers:
            case 1:
                self.model.add(Dense(1, input_shape=(m,)))
            case 2:
                self.model.add(Dense(n_neurons[0], activation=activations[0], input_shape=(m,)))
                self.model.add(Dense(1))
            case 3:
                self.model.add(Dense(n_neurons[0], activation=activations[0], input_shape=(m,)))
                self.model.add(Dense(n_neurons[1], activation=activations[1]))
                self.model.add(Dense(1))

        # Compilar el modelo
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=custom_optimizer, loss='mean_absolute_error', metrics=[metric])

    def fit(self, x_train, y_train, epochs=None, batch_size=None, validation_data=None):
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)


def filter(df):
    df.drop(columns=['Unnamed: 0'], inplace = True)

    df.sort_values(by=['Date', 'Location'], inplace = True)
    df.reset_index(drop=True, inplace=True)

    # Filtrado de lugares que son de interés
    locations = ["Sydney", "SydneyAirport", "Canberra", "Melbourne", "MelbourneAirport"]
    df = df[df['Location'].isin(locations)]
    df.reset_index(drop=True, inplace=True)

    df = df.drop(columns=["Location"])

    return df

def datecolumn2dateformat(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def add_day_number(df):
    #Creamos columna que indique el tiempo pasado
    #elegimos un día random como referencia:
    reference_date = pd.to_datetime('2012-12-12')
    df['DayNumber'] = (df['Date'] - reference_date).dt.days
    return df

def add_trig_date(df):
    # Para una mayor interpretabilidad, rotamos el calendario dejando la mitad del verano en los -90°
    day_of_midsummer = 30 + 15 - 10 # mes + medio mes - 10 días del final de diciembre

    rotated_angle_of_the_year = (df.Date.dt.day_of_year - day_of_midsummer) * 2 * np.pi / 365.25

    df['Autumness'] = (np.sin(rotated_angle_of_the_year) + 1) / 2
    df['Summerness'] = (np.cos(rotated_angle_of_the_year) + 1) / 2
    return df

def drop_normal_date(df):
    return df.drop(columns=["Date"])

def dirspeed2velxy(dire, speed):
    if pd.isna(dire) or pd.isna(speed):
        return np.nan, np.nan

    angle = dir2ang[dire]
    sin = np.sin(np.radians(angle))
    cos = np.cos(np.radians(angle))
    vely = sin * speed
    velx = cos * speed
    return velx, vely

def add_velxy(df):
    # Create new columns for sine and cosine of wind direction angles
    #9am
    aux = df.apply(lambda row: dirspeed2velxy(row['WindDir9am'], row['WindSpeed9am']), axis=1)
    df[['Wind9amVelX', 'Wind9amVelY']] = pd.Series(zip(*aux))

    #3pm
    aux = df.apply(lambda row: dirspeed2velxy(row['WindDir3pm'], row['WindSpeed3pm']), axis=1)
    df[['Wind3pmVelX', 'Wind3pmVelY']] = pd.Series(zip(*aux))

    #gust
    aux = df.apply(lambda row: dirspeed2velxy(row['WindGustDir'], row['WindGustSpeed']), axis=1)
    df[['WindGustVelX', 'WindGustVelY']] = pd.Series(zip(*aux))

    return df

def drop_og_wind(df):
    return df.drop(columns=['WindDir9am', 'WindDir3pm', 'WindGustDir', 'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed'])

def yn2bin(df):
    df.RainToday = df.RainToday.replace({'No': 0, 'Yes': 1})
    df.RainTomorrow = df.RainTomorrow.replace({'No': 0, 'Yes': 1})
    return df

def feature_selection(df):
    return df.drop(columns=['RainToday'])

def drop_nan_targets(df):
    df = df.dropna(subset=['RainfallTomorrow'])
    df.reset_index(drop=True)
    return df

def x_ys_splitter(df):
    # Column names for the transformations
    target_columns = ['RainfallTomorrow', 'RainTomorrow']
    columns_to_scale = df.drop(columns=target_columns).columns
    df_x = df.drop(columns=target_columns)
    df_y_reg, df_y_class = df[target_columns[0]], df[target_columns[1]]
    return df_x, df_y_reg, df_y_class

def f1_avg(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')
