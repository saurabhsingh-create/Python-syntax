# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:44:52 2020

@author: Saurabh.Singh
"""

#clean data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import r2score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook

from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

DATAPATH = 'Downloads/stock_prices_sample.csv'

#get crrent working directory
import os
os.getcwd()

#read file and save Date as index
data = pd.read_csv(DATAPATH, index_col=['DATE'], parse_dates=['DATE'])
data.head()

#number of rows and columns
data.shape

#get data types of all columns in data
data.dtypes

#filtering rows
data =  data[data.TICKER != 'GEF']
data = data[data.TYPE != 'Intraday']

#defining a list to drop columns

drop_cols = ['SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 'ADJ_VOLUME', 'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 'ADJ_OPEN', 'VOLUME', 'FREQUENCY', 'TYPE', 'FIGI']

data.drop(drop_cols, axis = 1, inplace = True)
data.head()
#to get list of columns
data.columns

#EDA

#plotting the closing stock (line chart)

plt.figure(figsize=(17, 8))
plt.plot(data.CLOSE)
plt.title('Closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.grid(False)
plt.show()


#Run moving average
#we first defing moving average function

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
#define rolling mean with window
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
    
    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
            
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')
    plt.grid(True)
    
#Smooth by the previous 5 days (by week)
plot_moving_average(data.CLOSE, 5)

#Smooth by the previous month (30 days)
plot_moving_average(data.CLOSE, 30)

#Smooth by previous quarter (90 days)
plot_moving_average(data.CLOSE, 90, plot_intervals=True)

#define exp smoothing
def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def plot_exponential_smoothing(series, alphas):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);

plot_exponential_smoothing(data.CLOSE, [0.05, 0.3])

#double exponential function
def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plot_double_exponential_smoothing(series, alphas, betas):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series.values, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)

plot_double_exponential_smoothing(data.CLOSE, alphas=[0.9, 0.02], betas=[0.9, 0.02])

#stationarity function
def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
tsplot(data.CLOSE, lags=30)

data_diff = data.CLOSE - data.CLOSE.shift(1)
#checking for stationarity
tsplot(data_diff[1:], lags=30)

#SARIMA
#Set initial values and some bounds
ps = range(0, 5)
d = 1
qs = range(0, 5)
Ps = range(0, 5)
D = 1
Qs = range(0, 5)
s = 5

#Create a list with all possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

def optimize_SARIMA(parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    
    results = []
    best_aic = float('inf')
    
    for param in tqdm_notebook(parameters_list):
        try: model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(param[0], d, param[1]),
                                               seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    #Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

result_table = optimize_SARIMA(parameters_list, d, D, s)


#Set parameters that give the lowest AIC (Akaike Information Criteria)

p, q, P, Q = result_table.parameters[0]

best_model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(p, d, q),
                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)

print(best_model.summary())

def plot_SARIMA(series, model, n_steps):
    """
        Plot model vs predicted values
        
        series - dataset with time series
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
    """
    
    data = series.copy().rename(columns = {'CLOSE': 'actual'})
    data['arima_model'] = model.fittedvalues
    #Make a shift on s+d steps, because these values were unobserved by the model due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    #Forecast on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    #Calculate error
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
    
    plt.figure(figsize=(17, 8))
    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1],alpha=0.5, color='lightgrey')
    plt.plot(data, label='actual')
    plt.legend()
    plt.grid(True);
    
# plot_SARIMA(data, best_model, 5)
print(best_model.predict(start=data.CLOSE.shape[0], end=data.CLOSE.shape[0] + 5))
print(mean_absolute_percentage_error(data.CLOSE[s+d:], best_model.fittedvalues[s+d:]))

comparison = pd.DataFrame({'actual': [18.93, 19.23, 19.08, 19.17, 19.11, 19.12],
                          'predicted': [18.96, 18.97, 18.96, 18.92, 18.94, 18.92]}, 
                          index = pd.date_range(start='2018-06-05', periods=6,))

comparison.head()

plt.figure(figsize=(17, 8))
plt.plot(comparison.actual)
plt.plot(comparison.predicted)
plt.title('Predicted closing price of New Germany Fund Inc (GF)')
plt.ylabel('Closing price ($)')
plt.xlabel('Trading day')
plt.legend(loc='best')
plt.grid(False)
plt.show()




#creating a dataframe
df =  pd.DataFrame(
        {"a" : [1,2,3],
         "b" : [4,5,6],
         "c" : [7,8,9]},
         index = [1,2,3])

df1 = pd.DataFrame(
        [[1,2,3],
         [4,5,6],
         [7,8,9]],
         index = [1,2,3],
         columns = ['a','b','c'])

#dataframe with multi indexes

df = pd.DataFrame(
        {"a" : [1,2,3],
         "b" : [4,5,6],
         "c" : [7,8,9]},
         index = pd.MultiIndex.from_tuples(
                 [('d',1), ('d', 2), ('d', 3)],
                 names = ['n', 'v']))

import pandas as pd
import numpy as np
print(np.__version__)
np.show_config()
np.info

import os
print(os.getcwd())

print(os.listdir(os.getcwd())

#create a null vector of size 10 but 5th indx as 1
Z= np.zeros(10)
Z[4] = 1

#create a vector with values in range 1 to 10
Z = np.arange(1,10)
Z[1]

#Reverse a vector
Z = np.arange(10)
Z = Z[::-1]

#create a 3*3 matrix with values ranging from 0 to 8
Z =np.arange(9).reshape(3,3)

#find indices of non zero elemnts in a list
Z = [1,1,0,0,0,1,0,0,0]
x =np.nonzero(Z)
print(x)

#create a 3*3 identity matrix
Z =np.eye(3)
print(Z)

#create a 3*3*3 array with random values
Z =  np.random.random((3,3,3))
Z =np.random.random((3))

#array of length 3
Z =np.random.random((3))

#array of length 3*3
Z =np.random.random((3))

Z.max()
Z.min()
Z.mean()

#create a vector of siz 30 and fil mean
Z =  np.random.random(30)
Z.mean()

#create a matrix with 1 one the border and zeros inside
Z = np.ones((4,4))
Z[1:-1,1:-1]=0
Z

#Create a 8x8 matrix and fill it with a checkerboard pattern
Z = np.zeros((8,8),dtype=int)
Z[1::2,::2] = 1
Z[::2,1::2] = 1
print(Z)
#Z = np.arange(10)
#Z[1::2,] #gives odd rows
#Z[::2,] #gives even rows
#Z[,::2] #even columns
#Z[,1:2] #gives off columns 

#normmaliz 5*5 random matrix
Z = np.random.random((5,5))
Zmax, Zmin= Z.max(), Z.min()
print(Zmax, Zmin)
nor = (Z-Zmin)/(Zmax-Zmin)


#get a dot product, or matrix multiplication
Z = np.dot(np.ones((5,3)), np.ones((3,2)))

#neagte all elements between 3 to 8
Z = np.arange(10)
Z[(Z>=3) & (Z<=8)] = -1

#sum up with range
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

#to round numbers
np.floor()
np.ceil()

#add matrix of 0 and valus
Z = np.zeros((5,5))
Z += np.arange(5)
print(Z)

def generate():
    for x in range(10):
        print(x)
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
#print(Z)
generate()

#create a vector and sort it
Z = np.arange(10)
Z.sort()
print(Z)

#sum variables
a =np.sum(P,Q)

#check if 2 arrays are qual
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)

#create a matrix and set write as False
Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1

#create a random var and replace max by 0
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

#to get min and max ebtween 2 numbers
np.minimum(10,2)
np.maximum(10,2)

#count most frequnet value in an array
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())

#to get n largest value from an array
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
# Slow
print (Z[np.argsort(Z)[-n:]])
# Fast
print (Z[np.argpartition(-Z,n)[:n]])





#K means clustering for loan prediction



#PROBLEM STATEMENT
#Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

#impport libraries
import pandas as pd
import numpy as no
import random as rd
import matplotlib.pyplot as plt


data = pd.read_csv("\\\\falmumapp43\\CP US NA Key Driver Analysis (20-SCP-3182)\\7. Personal Work Folder\\Saurabh\\Loan prediction\\clustering.csv")

data.head(10)

data.columns
#we have following columns
#['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
 #      'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
  #     'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
# we will choose loan amount and salary

X = data[["ApplicantIncome", "LoanAmount"]]

#plot the scatter plot
plt.scatter(X.ApplicantIncome, X.LoanAmount, c= "blue")
plt.xlabel("income")
plt.ylabel("LoanAmt")
plt.show()

X = data[["LoanAmount","ApplicantIncome"]]
#Visualise data points
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()

#here we choose random no of clusters and their centroids.

#define nuber of clusters
K= 3

#random obs as centroids
Centroids =  (X.sample(n=K))
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='blue')
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()


#assign all points to nearest clusters and recompure clusters
import numpy as np

diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["ApplicantIncome"]-row_d["ApplicantIncome"])**2
            d2=(row_c["LoanAmount"]-row_d["LoanAmount"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
    
    
#Plot scater of plot of new clusters
    
color=['blue','green','cyan']
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])
plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')
plt.xlabel('Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.show()


#End of loan prediction




#Problem Statement2 
#The goal of the problem statemnt is ustomer segmnetation based on thier annual spend on groeries, milk, etc.
#here instead of doing it manually we will use Kmeans library from sklearn.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

data =  pd.read_csv("\\\\falmumapp43\\CP US NA Key Driver Analysis (20-SCP-3182)\\7. Personal Work Folder\\Saurabh\\CustomerSegment\\Wholesale customers data.csv")

data.head()
data.shape  #dimension of the data
z = data.describe()  #similar to summarize function in r. describes stats of the data
#for ontinuous varaibles there is  a lot of vriation in data. therefore we need to scale it, since k means is distance based algorithm.
#we will use standard scaler as it gets the data in normalized form with mean 0 and sd 1.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

y = pd.DataFrame(data_scaled).describe()
#type(data)

#now the magnitude of all variables is similar
#define the kmeans function and initialize with kmeans++
#fit the data on sclaed data

kmeans = KMeans(n_clusters =2, init =  'k-means++', max_iter=100)
kmeans.fit(data_scaled)

#kmeans ++ produces better results as compared to random k.
#find out the indertia in fitted data, hwihc is intra cluster distance

kmeans.inertia_

#inertia can be plotted to see which clusters gives better result

#fitting muliple models

SSE =[]    

for i in range(1,20):
    kmeans = KMeans(n_clusters =i, init =  'k-means++', max_iter=100)
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
SSE

#convert the result in df and plot them
df =  pd.DataFrame({"Cluster" : range(1,20), "SSE" : SSE})
plt.plot(df.SSE, df.Cluster, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


#now by plot any cluster size between 5 to 8 is good
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

#to get the count of point in each cluster


frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()




#MALL data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
%matplotlib inline

data =  pd.read_csv("\\\\falmumapp43\\CP US NA Key Driver Analysis (20-SCP-3182)\\7. Personal Work Folder\\Saurabh\\CustomerSegment\\Mall_Customers.csv")

data.head()
data.shape  #dimension of the data
z = data.describe()  #similar to summarize function in r. describes stats of the data

data.drop(["Customer ID"], axis =1)
data.columns
df = data[['Gender','Age', 'Annual Income (k$)','Spending Score (1-100)' ]]

sns.boxplot(y=df["Spending Score (1-100)"], color="red")
sns.boxplot(y=df["Annual Income (k$)"])
sns.barplot(x=genders.index, y=genders.values)

from sklearn.cluster import KMeans

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

km = KMeans(n_clusters =5)
clusters = km.fit_predict(df.iloc[:,1:])    
df['Label']= clusters

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


#Corona vrus
pip install google-cloud 
pip install google-cloud-vision
pip install --upgrade google-api-python-client
pip install --user google-colab
import pandas as pd
#google colab is used to access files from google drive.
from google.colab import drive

pip install --user ConfigParser
drive.mount('content/drive')









