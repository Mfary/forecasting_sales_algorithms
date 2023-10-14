import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm




class KNNRegressor:
    def __init__(self, train_data, test_data):
        self.train_csv = pd.read_csv(train_data)
        self.pred_csv = pd.read_csv(test_data)
        
        
    def ready_monthly_train(self):
        self.train_csv = self.train_csv.drop(['store', 'item'], axis=1)
        self.train_csv['date'] = pd.to_datetime(self.train_csv['date'])
        self.train_csv['date'] = self.train_csv['date'].dt.to_period('M')     
        
        self.monthly_sales = self.train_csv.groupby('date').sum().reset_index()
        self.monthly_sales['date'] = self.monthly_sales['date'].dt.to_timestamp()
        
        result = adfuller(self.monthly_sales['sales'].dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        
        
        
    def arima_forecasting(self):
        model = pm.auto_arima(np.asarray(self.monthly_sales['sales']), start_p=1, start_q=1, max_p=3, max_q=3, test='adf', d=None, m=1, seasonal=False, start_P=0,
                      D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        
        # print(model.order)
        self.model = ARIMA(np.asarray(self.monthly_sales['sales']), order=model.order)
        self.fitted = self.model.fit()
        
        print(self.fitted.summary())
        
        
        
    
    
    def train(self):
        self.ready_monthly_train()
        self.arima_forecasting()
        
        
        
        
    def ready_monthly_pred(self):
        self.pred_csv = self.pred_csv.drop(['store', 'item', 'id'], axis=1)
        self.pred_csv['date'] = pd.to_datetime(self.pred_csv['date'])
        self.pred_csv['date'] = self.pred_csv['date'].dt.to_period('M')     
        
        self.monthly_sales_pred = self.pred_csv.groupby('date').sum().reset_index()
        self.monthly_sales_pred['date'] = self.monthly_sales_pred['date'].dt.to_timestamp()
        
        self.monthly_sales_pred['sales'] = 0
        self.monthly_sales_pred['diff_sales'] = 0
        
        self.monthly_total = pd.concat([self.monthly_sales,self.monthly_sales_pred]).reset_index(drop=True)
        # self.monthly_total = self.monthly_total.dropna()
        # print(self.monthly_total)
          
    
    
    
    def predict(self):
        self.ready_monthly_pred()
        
        # Forecast
        y = self.fitted.forecast(3, alpha=0.05)
        print(y)  # 95% conf
        print(type(y))
        self.monthly_total['sales'][-3:] = y
        print(self.monthly_total)

     
        
        
r = KNNRegressor("./train.csv", "./test.csv")
r.train()
r.predict()
r.monthly_total['diff_sales'] = r.monthly_total['sales'].diff()
r.monthly_total.to_csv('result/result_arima.csv')