import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import ast
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score, precision_score, classification_report, confusion_matrix, log_loss, recall_score
# 1. Загрузить данные из файла в формате CSV с помощью pandas.
data = pd.read_csv('result_learyted.csv')
X = data[['word_vectors']]
y = data['ozenka']

# 2. Разделить данные на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array([ast.literal_eval(x) for x in X_train['word_vectors']])
X_test = np.array([ast.literal_eval(x) for x in X_test['word_vectors']])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

# 5. Создать модель метода опорных векторов с линейным ядром.
import time 
t0 = time.time()

svm_model = SVC(kernel='linear')

# 6. Обучить модель на обучающих данных.
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

t1 = time.time() - t0
print("TIME = ", t1)

cm = confusion_matrix(y_test, y_pred)
specificity = cm[0,0]/(cm[0,1]+cm[0,0])
print(cm)
print("specificity ", specificity)

print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print("precision", precision)

report = classification_report(y_test, y_pred, target_names=['what', 'what2'])
print("report", report)

#print("log_loss ", log_loss(y_test, svm_model.predict_proba(X_test_scaled)))

recall = recall_score(y_test, y_pred)
print("recall ", recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='weighted')
print("f1", f1)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, svm_model.predict(X_test_scaled))
fpr, tpr, thresholds = roc_curve(y_test, svm_model.predict_proba(X_test_scaled)[:,1])
area = np.trapz(tpr, fpr)
print("area roc_auc", area)

test_data = pd.read_csv('some_learyted.csv', sep=',')
X_totest = test_data[['word_vectors']]
X_totest = np.array([ast.literal_eval(x) for x in X_totest['word_vectors']])
X_totest_scaled = min_max_scaler.transform(X_totest)

positive_count = 0
negative_count = 0

    # Определение тональности комментариев с помощью модели

sentiments = svm_model.predict(X_totest_scaled)
for sentiment in sentiments:
    if sentiment == 1:
        positive_count += 1
    elif sentiment == 0:
        negative_count += 1 
        
total_count = len(test_data)
positive_percent = (positive_count / total_count) * 100
negative_percent = (negative_count / total_count) * 100
print(f'Процент положительных комментариев: {positive_percent:.2f}%')
print(f'Процент отрицательных комментариев: {negative_percent:.2f}%')

  