######## A Healthcare Domain Chatbot to simulate the predictions of a General Physician ########
######## A pragmatic Approach for Diagnosis ############

# Importing the libraries
# from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

# Importing the dataset
training_dataset = pd.read_csv('Training.csv')
test_dataset = pd.read_csv('Testing.csv')

# Slicing and Dicing the dataset to separate features from predictions
X = training_dataset.iloc[:, 0:132].values
#print(X)
y = training_dataset.iloc[:, -1].values
#print(y)

# Dimensionality Reduction for removing redundancies
dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()
#print(dimensionality_reduction)

# Encoding String values to integer constants
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
#print(y)

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1, random_state = 2)

# Implementing the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Saving the information of columns
cols     = training_dataset.columns
cols     = cols[:-1]


# Checking the Important features
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Implementing the Visual Tree
from sklearn.tree import _tree

# This section of code to be run after scraping the data




# print(y_test)
model = LinearRegression()
model.fit(X_train, y_train)
r2_score=model.score(X_test,y_test)
print(r2_score*100)
# Execute the bot and see it in Action




# execute_bot()














































