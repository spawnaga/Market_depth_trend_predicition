import datetime
import sqlite3
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout, MaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


class ModelTraining:
    def __init__(self, database_path):

        self.db = sqlite3.connect(database_path)
        self.df = pd.read_sql_query("SELECT * from ES_market_depth", self.db)
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        self.scaler = MinMaxScaler()
        self.logdir = "logs/fit/" + "agentLearning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)

    def preprocess_data(self):
        self.df['difference'] = self.df['lastPrice'].shift(1) - self.df['lastPrice']
        self.df.dropna(inplace=True)
        self.df['direction'] = self.df['difference'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        self.X = self.df.iloc[:, 1:-6].values
        self.y = to_categorical(self.df['direction'].to_numpy(), num_classes=3)

    def train_test_split(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=random_state)
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def build_model(self):
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
        output = Dense(self.y.shape[1], activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    def train_model(self):
        trainHistory = self.model.fit(self.X_train, self.y_train, epochs=200000, batch_size=500,
                                      validation_data=(self.X_test, self.y_test),
                                      callbacks=[self.early_stop, self.tensorboard_callback])
        return trainHistory

if __name__ == '__main__':
    mt = ModelTraining(database_path=r"C:\Udemy\Interactive Brokers Python API\streaming ES\ES_ticks1.db")
    mt.preprocess_data()
    mt.train_test_split()
    mt.build_model()
    history = mt.train_model()
