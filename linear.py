import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LinearRegressor:
    def __init__(self, train_data, test_data):
        self.train_csv = pd.read_csv(train_data)
        self.pred_csv = pd.read_csv(test_data)
        
        
    def ready_monthly_train(self):
        self.train_csv = self.train_csv.drop(['store', 'item'], axis=1)
        self.train_csv['date'] = pd.to_datetime(self.train_csv['date'])
        self.train_csv['date'] = self.train_csv['date'].dt.to_period('M')     
        
        self.monthly_sales = self.train_csv.groupby('date').sum().reset_index()
        self.monthly_sales['date'] = self.monthly_sales['date'].dt.to_timestamp()
        
        self.monthly_sales['diff_sales'] = self.monthly_sales['sales'].diff()
        # self.monthly_sales = self.monthly_sales.dropna()
        
        
    def ready_supervised_data(self):
        self.supervised_data = self.monthly_sales.drop(['date', 'sales'], axis=1).dropna()
        for i in range(1,13):
            col = 'diff_' + str(i)
            self.supervised_data[col] = self.supervised_data['diff_sales'].shift(i)
        self.supervised_data = self.supervised_data.dropna().reset_index(drop=True)
        
        
    def split_test_train_data(self):
        self.train_data = self.supervised_data
        self.test_data = self.supervised_data[-12:]
        
    
    def scale_data(self):
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)
        self.scaler = scaler
        
    
    def preprocess_data(self):
        self.x_train, self.y_train = self.train_data[:,1:], self.train_data[:,0:1]
        self.x_test, self.y_test = self.test_data[:,1:], self.test_data[:,0:1]
        
        self.y_test = self.y_test.ravel()
        self.y_train = self.y_train.ravel()
        
        self.predict_df = pd.DataFrame(self.monthly_sales['date'][-12:].reset_index(drop=True))
        self.act_sales = self.monthly_sales['sales'][-13:].to_list()
        
        
    def linear_regression_forecasting(self):
        self.linreg = LinearRegression()
        self.linreg.fit(self.x_train, self.y_train)
        linreg_pred = self.linreg.predict(self.x_test)
        
        linreg_pred = linreg_pred.reshape(-1,1)
        linreg_pred_test_set = np.concatenate([linreg_pred,self.x_test], axis=1)
        linreg_pred_test_set = self.scaler.inverse_transform(linreg_pred_test_set)
        
        result_list = []
        for index in range(0, len(linreg_pred_test_set)):
            result_list.append(linreg_pred_test_set[index][0] + self.act_sales[index])
        linreg_pred_series = pd.Series(result_list,name='linreg_pred')
        self.predict_df = self.predict_df.merge(linreg_pred_series, left_index=True, right_index=True)
        
        linreg_rmse = np.sqrt(mean_squared_error(self.predict_df['linreg_pred'], self.monthly_sales['sales'][-12:]))
        linreg_mae = mean_absolute_error(self.predict_df['linreg_pred'], self.monthly_sales['sales'][-12:])
        linreg_r2 = r2_score(self.predict_df['linreg_pred'], self.monthly_sales['sales'][-12:])
        print('Linear Regression RMSE: ', linreg_rmse)
        print('Linear Regression MAE: ', linreg_mae)
        print('Linear Regression R2 Score: ', linreg_r2)
        
        
        
        plt.figure(figsize=(15,7))
        plt.plot(self.monthly_sales['date'], self.monthly_sales['sales'])
        plt.plot(self.predict_df['date'], self.predict_df['linreg_pred'])
        
        
    
    
    def train(self):
        self.ready_monthly_train()
        self.ready_supervised_data()
        self.split_test_train_data()
        self.split_test_train_data()
        self.scale_data()
        self.preprocess_data()
        self.linear_regression_forecasting()



    
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
        

    
    def ready_pred_data(self):
        self.monthly_total['diff_sales'] = self.monthly_total['sales'].diff()
        self.supervised_pred = self.monthly_total.drop(['date','sales'], axis=1).dropna()
        for i in range(1,13):
            col = 'diff_' + str(i)
            self.supervised_pred[col] = self.supervised_pred['diff_sales'].shift(i)
        self.supervised_pred = self.supervised_pred.dropna().reset_index(drop=True)
        # print(self.supervised_pred)
        
    
    
    
    
    def predict(self):
        self.ready_monthly_pred()
        
        for i in range (len(self.monthly_sales_pred.index)):
            self.ready_pred_data()
            self.pred_data = self.scaler.transform(self.supervised_pred)
            x = self.pred_data[-(len(self.monthly_sales_pred.index)):,1:]
    
            linreg_pred = self.linreg.predict(x)

            linreg_pred = linreg_pred.reshape(-1,1)
            linreg_pred_test_set = np.concatenate([linreg_pred,x], axis=1)
            linreg_pred_test_set = self.scaler.inverse_transform(linreg_pred_test_set)
            
            act_sales = self.monthly_total['sales'][-(len(self.monthly_sales_pred.index)+1):].to_list()
            
            result_list = []
            for index in range(0, len(linreg_pred_test_set)):
                result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
            # print(result_list)

            # y = linreg_pred_test_set[:,0]
            
            self.monthly_total.loc[len(self.monthly_total.index)-(len(self.monthly_sales_pred.index) - i), 'sales'] = result_list[i]
            # print(y)
            # print(self.monthly_total)
        plt.plot(self.monthly_total['date'][-(len(self.monthly_sales_pred.index)):], self.monthly_total['sales'][-(len(self.monthly_sales_pred.index)):])
        
        
        
        
        
r = LinearRegressor("./train.csv", "./test.csv")
r.train()
r.predict()
print(r.monthly_total)
r.monthly_total.to_csv('result/result_linear.csv')
plt.title("Customer Sales Forecast using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales", "Future sales"])
plt.show()