import datetime
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, MaxPool1D, LSTM, RNN, SimpleRNN, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import math


class ModelTraining:
    def __init__(self, database_path):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.model = None
        self.db = sqlite3.connect(database_path)
        self.df = pd.read_sql_query("SELECT * from ES_market_depth", self.db)
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        self.scaler = MinMaxScaler()
        self.logdir = "logs/fit/" + "agentLearning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.optimizer = Adam(learning_rate=0.01)
        self.lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * (0.1 ** (epoch // 10)))
        self.models = {}

    def preprocess_data(self):
        self.df['difference'] = self.df['lastPrice'].shift(1) - self.df['lastPrice']
        self.df.dropna(inplace=True)
        self.df['direction'] = self.df['difference'].apply(lambda x: 2 if x > 0 else 0 if x < 0 else 1)
        self.X = self.df.iloc[:, 1:-3].values
        self.y = self.df.loc[:, 'direction'].values

    def train_test_split(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=False)
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def create_models(self):
        self.models = {
            "Dense": self.Dense_model,
            "CNN": self.Dense_model,
            "LSTM": self.Dense_model,
            "RNN": self.Dense_model,
            "SimpleRNN": self.Dense_model,
            "CNNRNN": self.Dense_model,
            "xgboost": self.Dense_model,
            "treeDecision": self.Dense_model,
            "LinearRegression": self.Dense_model,
            "LogisticRegression": self.Dense_model,
            "KNeighborsClassifier": self.Dense_model,
            "RandomForestClassifier": self.Dense_model,
            "GradientBoostingClassifier": self.Dense_model,
            "SVC": self.Dense_model,
            "GaussianNB": self.Dense_model,
            "Perceptron": self.Dense_model,
            "MLPClassifier": self.Dense_model
        }

    def Dense_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1],))
        x = Dense(128, activation='relu')(input_layer)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def CNN_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1], 1))
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def LSTM_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1], 1))
        x = LSTM(units=64, return_sequences=True)(input_layer)
        x = Dropout(0.5)(x)
        x = LSTM(units=32, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = LSTM(units=16)(x)
        x = Dropout(0.5)(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def RNN_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1], 1))
        x = RNN(units=32, return_sequences=True)(input_layer)
        x = Dropout(0.5)(x)
        x = RNN(units=64, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def SimpleRNN_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1], 1))
        x = SimpleRNN(units=32, return_sequences=True)(input_layer)
        x = Dropout(0.5)(x)
        x = SimpleRNN(units=64, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def CNNRNN_model(self):
        self.y_train = to_categorical(self.y_train, num_classes=3)
        self.y_test = to_categorical(self.y_test, num_classes=3)
        input_layer = Input(shape=(self.X.shape[1], 1))
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool1D(pool_size=2)(x)
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Reshape((x.shape[1], 1))(x)
        x = SimpleRNN(units=64, activation='relu')(x)
        output = Dense(self.y_train.shape[1], activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        trainHistory = model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                 validation_data=(self.X_test, self.y_test),
                                 callbacks=[self.early_stop, self.tensorboard_callback, self.lr_schedule])
        return trainHistory.history['val_accuracy'][-1], trainHistory.history['val_loss'][-1]

    def xgboost_model(self):
        self.model = XGBClassifier(use_label_encoder=False)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def treeDecision_model(self):
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def LinearRegression_model(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def LogisticRegression_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def KNeighborsClassifier_model(self):
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def GradientBoostingClassifier_model(self):
        self.model = GradientBoostingClassifier(n_estimators=100)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def GaussianNB_model(self):
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def Perceptron_model(self):
        self.model = Perceptron()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss

    def MLPClassifier_model(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        loss = log_loss(self.y_test, y_pred, labels=y_pred)
        return accuracy, loss


if __name__ == '__main__':
    mt = ModelTraining(database_path=r".\CL_ticks.db")
    mt.preprocess_data()
    mt.train_test_split()
    accuracy, loss = mt.CNNRNN_model()
    print(accuracy, loss)
