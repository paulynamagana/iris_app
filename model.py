# Importing necessary libraries
import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.linear_model import LogisticRegression # Classification model

# Importing the datasetpip
data = pd.read_csv('./iris.csv')
print(data)

#Dictionary contatining the mapping
variety_mappings= {0:"Setosa", 1:"Versicolor", 2:"Virginica"}
data = data.replace(["Setosa", "Versicolor", "Virginica"], [0,1,2])
print(data)

#separate data
X = data.iloc[:,0:-1] # features/independent variables
y= data.iloc[:,-1] #target/dependent variable

logreg = LogisticRegression(max_iter=1000) #initializing te logistic regression model
logreg.fit(X,y)

def classify(a,b,c,d):
    arr = np.array([a,b,c,d]) #convert to numpy array
    arr = arr.astype(np.float64) #change data type to float
    query = arr.reshape(1,-1) #reshape the array
    prediction = variety_mappings[logreg.predict(query)[0]] #retrieve
    return prediction #return the prediction