# # importing required libraries
# import numpy as np
# import pandas as pd
# import pickle
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier


# # loading and reading the dataset

# heart = pd.read_csv("heart_cleveland_upload.csv")

# # creating a copy of dataset so that will not affect our original dataset.
# heart_df = heart.copy()

# # Renaming some of the columns 
# heart_df = heart_df.rename(columns={'condition':'target'})
# print(heart_df.head())

# # model building 

# #fixing our data in x and y. Here y contains target data and X contains rest all the features.
# x= heart_df.drop(columns= 'target')
# y= heart_df.target

# # splitting our dataset into training and testing for this we will use train_test_split library.
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

# #feature scaling
# scaler= StandardScaler()
# x_train_scaler= scaler.fit_transform(x_train)
# x_test_scaler= scaler.fit_transform(x_test)

# # creating K-Nearest-Neighbor classifier
# model=RandomForestClassifier(n_estimators=20)
# model.fit(x_train_scaler, y_train)
# y_pred= model.predict(x_test_scaler)
# p = model.score(x_test_scaler,y_test)
# print(p)

# print('Classification Report\n', classification_report(y_test, y_pred))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # Creating a pickle file for the classifier
# filename = 'heart-disease-prediction-knn-model.pkl'
# pickle.dump(model, open(filename, 'wb'))



import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
heart = pd.read_csv("heart_cleveland_upload.csv")
heart_df = heart.rename(columns={'condition': 'target'})

# Splitting data into features and target
X = heart_df.drop(columns='target')
y = heart_df['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save the model and scaler
pickle.dump(model, open('heart_disease_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))