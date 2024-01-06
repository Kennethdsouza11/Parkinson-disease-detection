#importing the dependecies
import numpy as np #usefull for making arrays
import pandas as pd #usefull for creating pandas dataframe which are structured data which make it easier to work on
from sklearn.model_selection import train_test_split #useful splitting the data into training and testing data
from sklearn.preprocessing import StandardScaler #useful for preprocessing the data before training it
from sklearn import svm #this is the model that we will be working on called the support vector machine
from sklearn.metrics import accuracy_score #used to evaluate our model's accuracy score and tell how good our model is. skleanr.metrics is a model which provides various classes and functions for checking the performance of our model

#data collection and analysis
#load the data from a csv file to a panda dataframe

parkinsons_data = pd.read_csv('parkinsons.csv') #pd.read_csv is a function in pandas library to read the data

#printing the first 5 rows of the dataframe
print(parkinsons_data.head())

#1 represents that person has parkinson and 0 represents that person doesnt have parkinson
                              
#number of rows and columns in the dataframe
print(parkinsons_data.shape) #this command will give us the number of rows and column in the dataframe in this dataframe we have 195 rows and 24 columns

#getting more info about the datset
print(parkinsons_data.info()) #gives extra info about the datset

#checking for missing values in each column
print(parkinsons_data.isnull().sum())

#if there are missing values then we need to create median or mean of those values and fill in the missing values before feeding into the training model

#getting some statistical measures about the data
parkinsons_data.describe()

#distribution of target variable (seeing how many have parkinson's and how many dont)
parkinsons_data['status'].value_counts() #1 -> parkinson's +
                                         # 0 -> healthy

#grouping the data based on the target value
parkinsons_data.groupby('status').mean() #here we will come to know the difference in value between the healthy and the person with parkinson's disease

#now we have analised our data and now we shall pre-preprocess it
#seperating the features and target (feature set are those set by which we make predictions and target set is the one we want to find the ans to it
X = parkinsons_data.drop(columns = ['name','status'],axis = 1) #when we are dropping a column we need to put axis is 1 and when we are dropping a row we need to put axis as 0. drop is used to drop scpecified labels(column or rows) from the dataframe

Y = parkinsons_data['status'] #this is the target which is the status column
print(X)

print(Y)

#splitting the data into training data and testing data
#creating four arrays X_train, X_test, Y_train and Y_test
#the corresponding labels of X_train will be stored in Y_train and the corresponding labels of Y_test will be stores in Y_test (labels here refer to the target values 0 and 1)

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2) #test_size = 0.2 means that we want 20% of the data as test data and 80% as training data and random_state is used to randomly split the dataset. we use this method becuase splitting the data will be different from computer to computer but if we want the computer to split the data in the same way then we use the same random_state value 

print(X.shape,X_train.shape, X_test.shape)

#Data standardisation that is we need to convert the values in a common range

scaler = StandardScaler()

scaler.fit(X_train) #used to understand the nature of the data and once it understands the nature of the data it will fit it without changing the nature of the data

X_train = scaler.transform(X_train) #transform method will transform the data in the same range

X_test = scaler.transform(X_test) # we dont need to fit the X_test data as it is a test for the model to check our accuracy later we just need to transform it in the same range

print(X_train)

#model training
#using support vector machine model

model = svm.SVC(kernel = 'linear') #SVC stands for support vector classifier. it classifies the data into classes (i.e the one with parkinsons and the one without parkinsons) SVR is support vector regression. example of SVC usage is when we want to predict if a person is male or female here we are going to classify the person as male or female. Example of SVR usage is when we want to predict the salary of a person based on its age, hours of work etc thus giving a particular discrete or continuous values

#training the SVM model with the training data
model.fit(X_train, Y_train) #this will fit our data points in the model and try to find the hyperplane

#model evaluation. here we will try to find the accuracy of our model
#checking the accuracy of X_train

X_train_pred = model.predict(X_train)
#it has now trained the model using SVM. X_train_pred are the values given by our model. now we need to compare it with the original values

training_data_acc = accuracy_score(Y_train, X_train_pred) #Y_train are the oringal values and X_train_pred are the values given by our model

print("Accuracy score of training data : ", training_data_acc)
# we got 88% accuracy score that means if we perform 100 tests then out of 100 tests 88 of them will be correct. Rule for good model is that its accuracy score should be above 75%

X_test_pred = model.predict(X_test)

test_data_acc = accuracy_score(Y_test, X_test_pred)
print("Accuracy score of test data : ", test_data_acc)

#Note our training data and test data accuracy's should almost be the same. if theres a large difference then it means our model has overtrained and this is called overfitting or underfitting one of the problems faced in training models

#building a predictive system that is using the model to predict new values

input_data = (202.26600,211.60400,197.07900,0.00180,0.000009,0.00093,0.00107,0.00278,0.00954,0.08500,0.00469,0.00606,0.00719,0.01407,0.00072,32.68400,0.368535,0.742133,-7.695734,0.178540,1.544609,0.056141) #created a input_data tuple

#changing the input data in a numpy array
input_data_as_numpy_array = np.asarray(input_data) #working with numpy array is more efficient. here its converting the input data from a tuple to a numpy array

#reshapping the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #the purpose of doing this is because intitally our X_train data set had 156 columns so when we pass in our input_data it has only 1 column so our model will wait for 156 columns to proceed hence we reshape it

#standardized the data
standard_data = scaler.transform(input_data_reshaped)

#to fit this in our model 
prediction = model.predict(standard_data)



if (prediction[0] == 0): #because the prediction is a list and [0] returns the first value
    print("The Person doesn't have parkinson disease")
else:
    print("The Person has parkinson disease")






