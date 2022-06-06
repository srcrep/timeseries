# Importing the libraries and setting up the working directory
import os
import zipfile
import pandas as pd
import pandas as pd
import os
import requests
from pymongo import MongoClient
import psycopg2
from sqlalchemy import create_engine
import psycopg2 
import io
import pandas.io.sql as sqlio
import requests
import json
import csv
import cv2
import pytesseract
import pandas as pd 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import lightgbm as lgb
import numpy as np
import pandas as pd
from fbprophet import Prophet

# Creating the function to download the files
def extract_data(file_name, file_path):
    #!kaggle competitions download  -d "NIFTY-50 Stock Market Data (2000 - 2020)" # -f $file_name -p $file_path --force
    !kaggle datasets download -d rohanrao/nifty50-stock-market-data
    
# file name
file_name = "NIFTY50_all.csv"

# file path
raw_data_path  = os.path.join(os.path.pardir, 'data', 'raw')
extract_data(file_name, raw_data_path)

# unzipping the downloaded file
import zipfile
with zipfile.ZipFile("C:/Users/Sony/Desktop/Python/nifty50-stock-market-data.zip","r") as zip_ref:
    zip_ref.extractall("C:/Users/Sony/stock_price/code")
    
# Changing the working directory
os.chdir('C:/Users/Sony/stock_price/code')

# Reading the file
import pandas as pd
df = pd.read_csv('NIFTY50_all.csv')

# Checking whether there are any null values 
df.isnull().sum()

# Dropping the columns that aren't required 
df.drop('Trades',axis=1,inplace=True)
df.drop('Deliverable Volume',axis=1,inplace=True)
df.drop('%Deliverble',axis=1,inplace=True)
df.drop('Turnover',axis=1,inplace=True)

#Converting dataframe to an csv file
csvfile = df.to_csv('NIFTY50_all2.csv', sep = ',',index=0,header=True)

#Creating a database in postgreSQL
try :
    dbConnection = psycopg2.connect(
        user = "postgres",
        password = "Login1-89",
        host = "localhost",
        port = "5432",
        database = "postgres")
    dbConnection.set_isolation_level(0)
    dbCursor = dbConnection.cursor()
    dbCursor.execute("CREATE DATABASE STOCKS1;")
    dbCursor.close()
except (Exception,psycopg2.Error) as dbError :
        print("Error while connecting to postgreSQL", dbError)
finally :
            if(dbConnection) :
                dbConnection.close()
                
#Creating the table in the postgresql
try:
    dbConnection = psycopg2.connect(
        user = "postgres",
        password = "Login1-89",
        host = "localhost",
        port = "5432",
        database = "stocks1")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute("""CREATE TABLE stock_stg(date text,
    Symbol text,
    Series text,
    Prev_Close float,
    Open float8,
    High float8,
    Low float8,
    Last float8,
    Close float8,
    VWAP float8,
    Volume integer);""")
    dbCursor.close()
    #dbCursor.execute("""ALTER TABLE Vehicles DROP COLUMN Total_registered_5""")
    #dbCursor.execute("""ALTER TABLE Vehicles ADD COLUMN Total_registered_5 INT""")
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection): dbConnection.close()

#Inserting the values in the postgre sql
import csv
try:
    dbConnection = psycopg2.connect(
        user="postgres",
        password="Login1-89",
        host="localhost",
        port="5432",
        database="stocks1")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    #insertString = "INSERT INTO Vehicles VALUES ('{}',"+"{},"*3+"{})"
    #with open("F:/NCI/DAP/DAP Code/DAP Code/raw_data/Sample/cleanedtwo.csv", 'r') as f:
    f = open(r'C:/Users/Sony/stock_price/code/NIFTY50_all2.csv', 'r')
    reader = csv.reader(f)
    next(reader) # skip the header
        #for row in reader:
         #   dbCursor.execute(insertString.format(*row))
    dbCursor.copy_from(f, 'stock_stg',sep=',')
    dbConnection.commit()
    dbCursor.close()
except (Exception , psycopg2.Error) as dbError :
    print ("Error:", dbError)
finally:
    if(dbConnection): dbConnection.close()
    
#reading the data
try:
    dbConnection = psycopg2.connect(
        user = "postgres",
        password = "Login1-89",
        host = "localhost",
        port = "5432",
        database = "stocks1")
    dbConnection.set_isolation_level(0) # AUTOCOMMIT
    dbCursor = dbConnection.cursor()
    dbCursor.execute("""SELECT * FROM stock_stg;""")
    dbCursor.close()
    #dbCursor.execute("""ALTER TABLE Vehicles DROP COLUMN Total_registered_5""")
    #dbCursor.execute("""ALTER TABLE Vehicles ADD COLUMN Total_registered_5 INT""")
except (Exception , psycopg2.Error) as dbError :
    print ("Error while connecting to PostgreSQL", dbError)
finally:
    if(dbConnection): dbConnection.close()  
    
#LOAD PSQL DATABASE


# Set up a connection to the postgres server.
conn_string = "host="+ "localhost" +" port="+ "5432" +" dbname="+ "stocks1" +" user=" + "postgres" \
+" password="+ "Login1-89"
conn=psycopg2.connect(conn_string)
print("Connected!")

# Create a cursor object
cursor = conn.cursor()


def load_data(value):

    sql_command = "SELECT * FROM stock_stg where Symbol like '%{0}%';".format(str(value))
    print (sql_command)

    # Load the data
    data = pd.read_sql(sql_command, conn)

    print(data.shape)
    return (data)    
    
# Text recognition
import cv2
import pytesseract
# read image
im = cv2.imread('F:/NCI/Research Project/bajaj.jpg')
# configurations
config = ('-l eng --oem 1 --psm 3')
# pytessercat
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Sony\AppData\Local\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
text = pytesseract.image_to_string(im, config=config)
# print text
text = text.split('\n')
#text[3]
text

df99 = text[1]
df99 = df99[0:4]
df99 = df99.upper()
data1 = load_data(df99)

data1

data = data1

data

#plot close price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices')
plt.plot(data['close'])
plt.title('BajajFinserv Inc. closing price')
plt.show()

# Plot the scatter plot
df_close = data['close']
df_close.plot(style='k.')
plt.title('Scatter plot of closing price')
plt.show()

df1 = data[['date','close']]

df1.isnull().sum()

#Describing the statistical information of the data
df1.close.describe()

#Determining the quantile (q1,q2)
q1 = df1.close.quantile(0.25)
q3 = df1.close.quantile(0.75)
q1,q3

# Determining the Inter Quartile Range
IQR = q3-q1
IQR

# Determining the lower limit and the upper limit
lower_limit = q1 - 1.5*IQR
upper_limit = q3 + 1.5*IQR
lower_limit, upper_limit

#Here are the outliers
df1[(df1.close<lower_limit)|(df1.close>upper_limit)]

# Data free from outliers are taken for analysis
df1_no_outlier = df1[(df1.close>lower_limit)&(df1.close<upper_limit)]
df1_no_outlier

# Convert Date into Datetime
df1_no_outlier['date']=pd.to_datetime(df1_no_outlier['date'])

# Indexing with respect to date column
df1_no_outlier.set_index('date',inplace=True)

# Determing the standard deviation and the mean value of the data
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()

df_close = df1_no_outlier['close']

#plot close price with no outliers
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Close Prices with no outliers')
plt.plot(data['close'])
plt.title('BajajFinserv Inc. closing price with no outliers')
plt.show()

#Seed for FB prophet model
myfavouritenumber = 37
seed = myfavouritenumber
np.random.seed(seed)

#Fitting the FB prophet model
model_fbp = Prophet()
for feature in exogenous_features:
    model_fbp.add_regressor(feature)

model_fbp.fit(df_train[["date", "close"] + exogenous_features].rename(columns={"date": "ds", "close": "y"}))

forecast = model_fbp.predict(df_valid[["date", "close"] + exogenous_features].rename(columns={"date": "ds"}))
df_valid["Forecast_Prophet"] = forecast.yhat.values

# Forecasting the data for EDA
model_fbp.plot_components(forecast)

test_result=adfuller(df1_no_outlier['close'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(df1_no_outlier['close'])

#Differencing of Close price

df1_no_outlier['Sales First Difference'] = df1_no_outlier['close'] - df1_no_outlier['close'].shift(1)

## Again test dickey fuller test
adfuller_test(df1_no_outlier['Sales First Difference'].dropna())

#plot close price with no outliers
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Sales First Difference')
plt.plot(df1_no_outlier['Sales First Difference'])
plt.title('First Level Sales Difference')
plt.show()

df1_no_outlier

df1_no_outlier.dropna(axis = 0, inplace = True)

# Plotting the auto correlation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df1_no_outlier['close'])
plt.show()

# Plotting the PACF and ACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df1_no_outlier['Sales First Difference'].dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df1_no_outlier['Sales First Difference'].dropna(),lags=40,ax=ax2)

# Splitting the data into train and test data
df1_train = df1_no_outlier[df1_no_outlier.index < "2019"]
df1_test = df1_no_outlier[df1_no_outlier.index >= "2019"]

df1_train

df1_train.shape

df1_test.shape

# Taking the log values of the train and test data
train_data, test_data = np.log(df1_train['close']), np.log(df1_test['close'])

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(train_data, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

train_data.shape, test_data.shape

# Sending the data to df6 for the ARIMAX model
df6 = data

# Convert Date into Datetime
df6['date']=pd.to_datetime(df6['date'])

# Setting the date column as index
df6.set_index("date", drop=False, inplace=True)
df6.head()

# Determining the lag features for the model by keeping the window for three days, 1 week and 1 month
df6.reset_index(drop=True, inplace=True)
lag_features = ["high", "low", "volume"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df6[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df6[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df6[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df6[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df6[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df6[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df6[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df6[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df6[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df6.fillna(df6.mean(), inplace=True)

df6.set_index("date", drop=False, inplace=True)
df6.head()

#Determing the other exogenous features
df6.date = pd.to_datetime(df6.date, format="%Y-%m-%d")
df6["month"] = df6.date.dt.month
df6["week"] = df6.date.dt.week
df6["day"] = df6.date.dt.day
df6["day_of_week"] = df6.date.dt.dayofweek
df6.head()

# Splitting the data into train and test data
df_train = df6[df6.date < "2019"]
df_valid = df6[df6.date >= "2019"]

# Assigning the lag values and other additional date time values to the exogenous feature data frame
exogenous_features = ["high_mean_lag3", "high_std_lag3", "low_mean_lag3", "low_std_lag3",
                      "volume_mean_lag3", "volume_std_lag3",
                      "high_mean_lag7", "high_std_lag7", "low_mean_lag7", "low_std_lag7",
                      "volume_mean_lag7", "volume_std_lag7","high_mean_lag30", "high_std_lag30", 
                      "low_mean_lag30", "low_std_lag30",
                      "volume_mean_lag30", "volume_std_lag30","month", "week", "day", "day_of_week"]

# Fitting the ARIMAX model
model = auto_arima(df_train.close, exogenous=df_train[exogenous_features], trace=True, error_action="ignore", suppress_warnings=True)
model.fit(df_train.close, exogenous=df_train[exogenous_features])

# Predicting the close price data for the test data
forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid["Forecast"] = forecast

aa_rmse = np.sqrt(mean_squared_error(df_valid.close, df_valid.Forecast))
aa_mae = mean_absolute_error(df_valid.close, df_valid.Forecast)
aa_rmse
aa_mae

# Evaluation metrics of the ARIMAX model
print("RMSE of Auto ARIMAX:", np.sqrt(mean_squared_error(df_valid.close, df_valid.Forecast)))
print("\nMAE of Auto ARIMAX:", mean_absolute_error(df_valid.close, df_valid.Forecast))

# Forecasting the close price through ARIMAX model
df_valid[["close", "Forecast"]].plot(figsize=(14, 7))

plt.plot(df_train.close, label='training')
plt.plot(df_valid.close, color = 'blue', label='Actual Stock Price')
plt.plot(df_valid.Forecast, color = 'orange',label='Predicted Stock Price')
#plt.fill_between(lower_series.index, lower_series, upper_series, 
#                 color='k', alpha=.10)


plt.title('BajajFinserv. Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()
