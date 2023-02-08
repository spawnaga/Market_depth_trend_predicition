# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:07:20 2020

@author: alial
"""


# import libraries

import sqlite3

import numpy as np
import pandas as pd
from IPython.display import display, clear_output
from ib_insync import *


def onTickerUpdate(ticker):
    global ticker_size
    global ticker_price
    ticker_size.append(ticker.lastSize)
    ticker_price.append(ticker.last)
    if len(ticker_size) > 20:
        ticker_size = [0,0]
    if len(ticker_price) > 20:
        ticker_price = [0,0]
    if ticker_size[-1] == ticker_size[-2] or ticker_price[-1] == ticker_price[-2]:
        return
    bids = ticker.domBids
    for i in range(5):
        df.iloc[i, 0] = bids[i].size if i < len(bids) else 0
        df.iloc[i, 1] = bids[i].price if i < len(bids) else 0
        
    asks = ticker.domAsks
    for i in range(5):
        df.iloc[i, 2] = asks[i].price if i < len(asks) else 0
        df.iloc[i, 3] = asks[i].size if i < len(asks) else 0
    last = ticker.domTicks[0].price
    print(df.columns)
    x= df.values.flatten()
    x = np.append(x, last)
    df_x.index = [ticker.time]
    print(x.shape, df_x.shape)
    df_x.iloc[0,0:] = x
    clear_output(wait=True)
    display(df)
    display(last)
    display(x)
    display(df_x)
    df_x.to_sql(name='ES_market_depth', con = db, if_exists='append')



util.startLoop() #required for ib_insync to use in kernals of jupyter and spyder
ib = IB() #initialize IB class for IB API Client and Wrapper async version
ib.disconnect() #to start a new connection disconnect older ones
ib.connect('127.0.0.1',7496,8) #connect to IB server
ib.portfolio()

ticker_size = [0,0] #initialize arrays for size 
ticker_price = [0,0] #initialize arrays for price 


contract=Future(conId=256019251, symbol='CL', lastTradeDateOrContractMonth='20211220', right='0', multiplier='1000', primaryExchange='NYMEX', currency='USD', localSymbol='CLF2', tradingClass='CL') #define contract

ib.qualifyContracts(contract) #qualify contract

ticker = ib.reqMktDepth(contract) #request market depth for the ticker
df = pd.DataFrame(index=range(5),
        columns='bidSize bidPrice askPrice askSize'.split()) #save the ticker in a df

df_x = pd.DataFrame(index = range(1), 
                    columns= 'bidSize_1 bidPrice_1 askPrice_1 askSize_1 \
                        bidSize_2 bidPrice_2 askPrice_2 askSize_2 bidSize_3 \
                            bidPrice_3 askPrice_3 askSize_3 bidSize_4 \
                                bidPrice_4 askPrice_4 askSize_4 bidSize_5 \
                                    bidPrice_5 askPrice_5 askSize_5 lastPrice'\
                                        .split()) #updating db raws 

    
""" initialize sql file """

db = sqlite3.connect(r'C:\Udemy\Interactive Brokers Python API\streaming ES\CL_ticks.db')
c=db.cursor()
c.execute('DROP TABLE IF EXISTS ES_market_depth')
# c.execute("CREATE TABLE ES_market_depth (time datetime primary key,bidSize_1 integer, bidPrice_1 real(15,5), askPrice_1 real(15,5), askSize_1 integer, bidSize_2 integer, bidPrice_2 real(15,5), askPrice_2 real(15,5), askSize_2 integer, bidSize_3 integer, bidPrice_3 real(15,5), askPrice_3 real(15,5), askSize_3 integer, bidSize_4 integer, bidPrice_4 real(15,5), askPrice_4 real(15,5), askSize_4 integer, bidSize_5 integer, bidPrice_5 real(15,5), askPrice_5 real(15,5), askSize_5 integer, lastPrice real(15,5))")
try:
    db.commit()
except:
    db.rollback()
    
# ib.disconnect()
    
    
    
ib.sleep(2)
ticker.updateEvent += onTickerUpdate

IB.sleep(15);