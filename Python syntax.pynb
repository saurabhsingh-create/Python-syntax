##########Python syntax


#PANDAS

Creating a dataframe
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


#Subset varaiabes / columns 

for multiple columns
df[['width', 'length']]

for single columns
df.width or df['width']

select all columns from columns A to D(indexes)
df.loc[:, 'A':'D']
Select columns by indexes
df.iloc[:, 1:5]
df.iloc[:,[1,2,3,4,5]]

Select multiple columns based on a condition on rows
df.loc[df['a']>0,['a','c']]

#Subset rows/Observations
Extract rows that meet a logical criteria
df[df.Length >0]
df[df['Length]>0]

#drop duplicate rows
df.drop_duplicates()

#Select 1st n rows
df.head(n)

#select last n rows
df.tail(n)

#view colums of a dataframe
df.columns

#randomly select fractaion of rows
df.sample(frac = 0.5)

#randomly select n rows
df.sample(n=10)

#select rows by position
df.iloc[10:20]

#Gather column into rows
pd.melt(df)

#append rows of 2 dataframes similar to rbind
pd.concat([df1,df2])

#append columns of 2 dataframes similar to cbind
pd.concat([df1,df2], axis = 1)

#spread rows into columns
df.pivot(columns = 'var', values = 'val')

#sort values by a a column in ascending order
df.sort_values('mpg')

#sort in descending order
df.sort_values('mpg', ascending = False)

#sort the index of a dataframe
df.sort_index()

#reset index to rownmbers mpving index to columns
df.reset_index()

#rename columns of a dataframe
df.rename(columns = {'cyl':'cycle'})

#drop multiple columns from a dataframe
df.drop(['Length', 'Height'], axis = 1)

#read a excel file 
df = pd.read_excel (r'Path where the Excel file is stored\File name.xlsx')
print (df)

#read a csv file
data = pd.read_csv("filename.csv") 

#import, get version, config
import numpy as np
print(np.__version__)
np.show_config()
np.info

# Find out your current working directory
import os
print(os.getcwd())

# Display all of the files found in your current working directory
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

#get mean, min, max from a vector
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

#normmaliz 5*5 random matrix. min max scaler
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

#to get dimension of a dataframe
df.shape
df.shape[0] --row count
df.shape[1] -- columns count

#plot the scatter plot
plt.scatter(X.ApplicantIncome, X.LoanAmount, c= "blue")
plt.xlabel("income")
plt.ylabel("LoanAmt")
plt.show()

#method 2 of slecting columns
plt.scatter(X["ApplicantIncome"],X["LoanAmount"],c='black')

#subset columns
X = data[["LoanAmount","ApplicantIncome"]]

#Choose random 3 points
Centroids =  (X.sample(n=3))

#to iterate over rows of a dataframe
for index, row_x in data.iterows():
	print(index) #gives indexes
	print(row_x)#givess data
	
#use range on loops
for i in range(K):
    print(i)

#groupby by and mean
Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]	

#import KMeans from sklearn
from sklearn.cluster import KMeans

%matplotlib inline makes your plot outputs appear and be stored within the notebook. It is a agic function

#to get statistics of a dataset
df.describe()  #gives count, mean, sd, min, 25, 50, 75, max

#line chart of 2 varaibles
plt.plot(df.SSE, df.Cluster, marker = 'o')

#run kmeans in loop function
SSE =[]    

for i in range(1,20):
    kmeans = KMeans(n_clusters =i, init =  'k-means++', max_iter=100)
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)
SSE

#to get intra cluster similarity in kmeans
kmeans.inertia_

#convert to a dataframe 
df =  pd.DataFrame({"Cluster" : range(1,20), "SSE" : SSE})

#apply a standard scaler to data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#define kmeans, fit and prdict
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

#to get count of data points in a cluster
frame['cluster'] = pred
frame['cluster'].value_counts()

#ploting a box plot
sns.boxplot(y=df["Spending Score (1-100)"], color="red")

#to get vaue count of occurences
genders = df.Gender.value_counts()

#barplot
sns.barplot(x=genders.index, y=genders.values)

#subset all rows but from 2nd columns
df.iloc[:,1:]

google.colab :: used to access files