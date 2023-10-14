import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly




class ProphetPredictor:
    def __init__(self, train_data, test_data):
        self.train_csv = pd.read_csv(train_data)
        self.pred_csv = pd.read_csv(test_data)
        
        
    def ready_monthly_train(self):
        self.train_csv['date'] = pd.to_datetime(self.train_csv['date'])
        train = self.train_csv.copy()
        train['store/item'] = train['store'].apply(str) + '-' + train['item'].apply(str) 
        self.train_data = train.drop(['item', 'store'], axis=1, inplace=False)
        
    def train(self):
        self.ready_monthly_train()
        results = []
        line_list = self.train_data['store/item'].unique()
        for line in line_list:
            df = self.train_data[self.train_data['store/item'] == line]
            df = df.drop(['store/item'], axis=1, inplace=False)
            df.columns = ['ds', 'y']
            
            model = Prophet(interval_width=0.95)
            model.fit(df)
            future = model.make_future_dataframe(periods=90)
            forcast = model.predict(future)
            forcast['store/item'] = line
            forcast = forcast[['store/item', 'ds', 'yhat']]
            # print(line)
            # p1 = model.plot(forcast)
            # plt.show()
            # p2 = model.plot_components(forcast)
            # plt.show()
            print(forcast)
            print('_______________________________________________________________________')
            
            


p = ProphetPredictor('train.csv', 'test.csv')
p.train()