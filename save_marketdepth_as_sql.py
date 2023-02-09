import sqlite3
import numpy as np
import pandas as pd
from ib_insync import *
import random
import os


class StreamData:
    def __init__(self, db_path="./CL_ticks.db"):
        self.ticker_size = [0, 0]
        self.ticker_price = [0, 0]
        self.df = pd.DataFrame(index=range(5), columns='bidSize bidPrice askPrice askSize'.split())
        self.df_x = pd.DataFrame(index=range(1), columns='bidSize_1 bidPrice_1 askPrice_1 askSize_1 \
                                bidSize_2 bidPrice_2 askPrice_2 askSize_2 bidSize_3 \
                                    bidPrice_3 askPrice_3 askSize_3 bidSize_4 \
                                        bidPrice_4 askPrice_4 askSize_4 bidSize_5 \
                                            bidPrice_5 askPrice_5 askSize_5 lastPrice'.split())
        self.last_price = None
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7496, random.randint(0, 100))
        self.contract = ContFuture(symbol="NQ", exchange="CME", currency="USD")
        self.ib.qualifyContracts(self.contract)
        self.ticker = self.ib.reqMktDepth(self.contract)
        self.db_file = db_path
        self.db = sqlite3.connect(self.db_file)
        if not os.path.exists(self.db_file):
            self.cursor = self.db.cursor()
            self.cursor.execute('DROP TABLE IF EXISTS ES_market_depth')
            self.db.commit()
        else:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()

    def start_stream(self):
        self.cursor.execute('DROP TABLE IF EXISTS ES_market_depth')
        self.db.commit()
        self.ticker.updateEvent += self.on_ticker_update
        self.ib.run()

    def on_ticker_update(self, ticker):
        bids = ticker.domBids
        for i in range(5):
            self.df.iloc[i, 0] = bids[i].size if i < len(bids) else 0
            self.df.iloc[i, 1] = bids[i].price if i < len(bids) else 0

        asks = ticker.domAsks
        for i in range(5):
            self.df.iloc[i, 2] = asks[i].price if i < len(asks) else 0
            self.df.iloc[i, 3] = asks[i].size if i < len(asks) else 0
        self.last_price = ticker.domTicks[0].price
        x = self.df.values.flatten()
        x = np.append(x, self.last_price)
        self.df_x.iloc[0, :] = x
        self.df_x.to_sql(name='ES_market_depth', con=self.db, if_exists='append')


def main():
    stream = StreamData('./CL_ticks.db')
    stream.start_stream()


if __name__ == "__main__":
    main()
