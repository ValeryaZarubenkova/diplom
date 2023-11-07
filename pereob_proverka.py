import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import sklearn.model_selection
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
products = pd.read_csv('result_learyted.csv', sep=',')

# Split data into training and testing sets
Kek = products[['word_vectors']]
y = products['ozenka']

test_data = pd.read_csv('some_learyted.csv', sep=',')
X_totrain = test_data[['word_vectors']]
y_totrain = test_data['ozenka']
print(len(Kek), len(y))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Kek, y, test_size=0.25)
# Convert string vectors to numpy arrays
X_train = np.array([ast.literal_eval(x) for x in X_train['word_vectors']])
X_test_2 = np.array([ast.literal_eval(x) for x in X_test['word_vectors']])
X_test = np.array([ast.literal_eval(x) for x in X_totrain['word_vectors']])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled_2 = scaler.transform(X_test_2)
X_test_scaled = scaler.transform(X_test)
# Define logistic regression model
model = sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000)

# Train model on training data
model = model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled_2, y_test)

print('Accuracy:', accuracy)

# Evaluate model on testing data
#accuracy = model.score(X_test_scaled, y_totrain)

#print('Accuracy:', accuracy)
y_pred = model.predict(X_test_scaled)



positive_count = 0
negative_count = 0
for sentiment in y_pred:
    if sentiment == 1:
        positive_count += 1
    elif sentiment == 0:
        negative_count += 1   
   # print("sentiment", sentiment)


# Вычисление процента положительных и отрицательных комментариев
total_count = len(test_data)
positive_percent = (positive_count / total_count) * 100
negative_percent = (negative_count / total_count) * 100

print(f'Процент положительных комментариев: {positive_percent:.2f}%')
print(f'Процент отрицательных комментариев: {negative_percent:.2f}%')




