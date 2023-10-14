import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMTrainer:
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
        
        
    def lstm_forecasting(self):
        x_train_lstm = self.x_train.reshape(self.x_train.shape[0], 1, self.x_train.shape[1])
        x_test_lstm = self.x_test.reshape(self.x_test.shape[0], 1, self.x_test.shape[1])
        self.lstm = Sequential()
        self.lstm.add(LSTM(4, batch_input_shape=(1, x_train_lstm.shape[1], x_test_lstm.shape[2])))
        self.lstm.add(Dense(100, activation='relu'))
        self.lstm.add(Dense(1))
        self.lstm.compile(optimizer='adam', loss='mean_squared_error')
        
        checkpoint_filepath = os.getcwd()
        model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
        callbacks = [EarlyStopping(patience=5), model_checkpoint_callback]
        
        history = self.lstm.fit(x_train_lstm, self.y_train, epochs=200, batch_size=1, validation_data=(x_test_lstm, self.y_test), callbacks=callbacks)

        metrics_df = pd.DataFrame(history.history)
        print(metrics_df)
        
        
        plt.figure(figsize=(10,5))
        plt.plot(metrics_df.index, metrics_df.loss)
        plt.plot(metrics_df.index, metrics_df.val_loss)
        plt.title('Sales Forecast Model Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.show()

        lstm_pred = self.lstm.predict(x_test_lstm, batch_size=1)
        
        lstm_pred = lstm_pred.reshape(-1,1)
        lstm_pred_test_set = np.concatenate([lstm_pred,self.x_test], axis=1)
        lstm_pred_test_set = self.scaler.inverse_transform(lstm_pred_test_set)
        
        result_list = []
        for index in range(0, len(lstm_pred_test_set)):
            result_list.append(lstm_pred_test_set[index][0] + self.act_sales[index])
        lstm_pred_series = pd.Series(result_list,name='lstm_pred')
        self.predict_df = self.predict_df.merge(lstm_pred_series, left_index=True, right_index=True)
        
        lstm_rmse = np.sqrt(mean_squared_error(self.predict_df['lstm_pred'], self.monthly_sales['sales'][-12:]))
        lstm_mae = mean_absolute_error(self.predict_df['lstm_pred'], self.monthly_sales['sales'][-12:])
        lstm_r2 = r2_score(self.predict_df['lstm_pred'], self.monthly_sales['sales'][-12:])
        print('LSTM RMSE: ', lstm_rmse)
        print('LSTM MAE: ', lstm_mae)
        print('LSTM R2 Score: ', lstm_r2)
        
        
        
        plt.figure(figsize=(15,7))
        plt.plot(self.monthly_sales['date'], self.monthly_sales['sales'])
        plt.plot(self.predict_df['date'], self.predict_df['lstm_pred'])
        
        
    
    
    def train(self):
        self.ready_monthly_train()
        self.ready_supervised_data()
        self.split_test_train_data()
        self.split_test_train_data()
        self.scale_data()
        self.preprocess_data()
        self.lstm_forecasting()



    
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
            
            x_test = x.reshape(x.shape[0], 1, x.shape[1])
    
            lstm_pred = self.lstm.predict(x_test, batch_size=1)

            lstm_pred = lstm_pred.reshape(-1,1)
            lstm_pred_test_set = np.concatenate([lstm_pred,x], axis=1)
            lstm_pred_test_set = self.scaler.inverse_transform(lstm_pred_test_set)
            
            act_sales = self.monthly_total['sales'][-(len(self.monthly_sales_pred.index)+1):].to_list()
            
            result_list = []
            for index in range(0, len(lstm_pred_test_set)):
                result_list.append(lstm_pred_test_set[index][0] + act_sales[index])
            # print(result_list)

            # y = linreg_pred_test_set[:,0]
            
            self.monthly_total.loc[len(self.monthly_total.index)-(len(self.monthly_sales_pred.index) - i), 'sales'] = result_list[i]
            # print(y)
            # print(self.monthly_total)
        plt.plot(self.monthly_total['date'][-(len(self.monthly_sales_pred.index)):], self.monthly_total['sales'][-(len(self.monthly_sales_pred.index)):])
        
        
        
        
            
r = LSTMTrainer("./train.csv", "./test.csv")
r.train()
r.predict()
print(r.monthly_total)
r.monthly_total.to_csv('result/result_ltms.csv')
plt.title("Customer Sales Forecast using LSTM")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales", "Future sales"])
plt.show()