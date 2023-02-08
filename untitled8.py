# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 2:38:29 2022

@author: alial
"""


""" updated from previous version. I followed suggestions from reddit members 
morphicon : suggested to remove shuffle in data distribution and use RNN, LSTM Or XGBoost
and Very_Large_Cone which suggested to not shuffle the data and split on a time-series
Very_Large_Cone: catched the mistake of shuffling data
"""

# imports
import sqlite3

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


#early stop class loded from stackoverflow link: https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='val_loss', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get(self.monitor)
        if val_loss is not None:
            if val_loss <= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
# The following lines are to setup my GPU for the learning
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
# End of GPU setup


db = sqlite3.connect(r'C:\Udemy\Interactive Brokers Python API\streaming ES\ES_ticks.db') # Loading the file
df = pd.read_sql_query("SELECT * from ES_market_depth", db) # converting file into a Dataframe
# df = pd.read_csv(r'C:\Projects\Market_depth_trend_predicition-main\MarketDepth_data_sample.csv',index_col='index',parse_dates=True)
df['difference'] = df['lastPrice'].shift(1)-df['lastPrice'] # setting the difference between two tics (current and last tick) and shifting one row up 
df.dropna(inplace = True) 

# setting the direction of the trend and separting into categories for learning

"""
same with a value of 1 means the trend is not changing
uptrend with a value of 1 means the trend is up
downtrend with a value of 1 means the trend is down
"""

df['difference'] = df['lastPrice'].shift(1)-df['lastPrice']
df.dropna(inplace = True)
df['direction'] = df['difference'].apply(lambda x: 1 if x>0 else -1 if x<0 else 0)


#Setting X and y
X = df.iloc[:,1:-6].values
y = to_categorical(df['direction'].to_numpy(),num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Normalizing data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



#setting up the model of tensorflow
input_layer = Input(shape=(X_train.shape[1],1))
x=input_layer
for _ in range(1): # five layers
    x = Dropout(0.2)(x) # Dropout to avoid overfitting
    x = CuDNNLSTM(X_train.shape[1], return_sequences = True)(x) # using LSTM with return sequences to adopt to time sequences
x = GlobalAveragePooling1D()(x) # Global averaging to one layer shape to feed to a dense categorigal classification
output = Dense(y.shape[1], activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
# 
#creating an early stop based on minmizing val_loss
# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000,restore_best_weights=True)
early_stop = EarlyStopping(monitor='val_loss', patience = 1) #[TerminateOnBaseline(monitor='val_loss', baseline=0.05)]

#fit the model
r = model.fit(X_train, y_train, epochs = 200000, batch_size=200,
             validation_data = (X_test, y_test), callbacks=[early_stop], shuffle=False,use_multiprocessing=True) #fit the data without shuffling
#plot the results.
pd.DataFrame(r.history).plot()

metrics = pd.DataFrame(model.history.history)
metrics.plot()
model.metrics_names
model.evaluate(X_test, y_test, verbose=0)

from sklearn.metrics import classification_report, confusion_matrix
predictions = (model.predict(X_test) > 0.5).astype("int32")

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
sns.heatmap(confusion_matrix(y_test,predictions))
