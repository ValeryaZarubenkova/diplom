from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import sklearn.model_selection
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss, recall_score
import pandas as pd
import numpy as np
import ast

from sklearn.model_selection import train_test_split
def f_ranforest():
    products = pd.read_csv('result_learyted.csv', sep=',')

    # Split data into training and testing sets
    X = products[['word_vectors']]
    y = products['ozenka']

    print(len(X), len(y))


    test_data = pd.read_csv('some_learyted.csv', sep=',')
    X_totest = test_data[['word_vectors']]
    


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
    # Convert string vectors to numpy arrays
    X_train = np.array([ast.literal_eval(x) for x in X_train['word_vectors']])
    X_test = np.array([ast.literal_eval(x) for x in X_test['word_vectors']])
    X_totest = np.array([ast.literal_eval(x) for x in X_totest['word_vectors']])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_totest_scaled = scaler.transform(X_totest)
    # Создаём модель леса из сотни деревьев
    import time 
    t0 = time.time()

    model2 = RandomForestClassifier(n_estimators=100,
                                bootstrap = True,
                                max_features = 'sqrt')
    # Обучаем на тренировочных данных
    model2.fit(X_train_scaled, y_train)

    # Действующая классификация
    y_pred = model2.predict(X_test_scaled)

    t1 = time.time() - t0
    print("TIME = ", t1)

    from sklearn.metrics import roc_auc_score

    # Рассчитываем roc auc
    roc_value = roc_auc_score(y_test, y_pred)
    print(roc_value)

    accuracy = model2.score(X_test_scaled, y_test)
    print('Accuracy:', accuracy)

    report = classification_report(y_test, y_pred, target_names=['what', 'what2'])
    print("report", report)
    print("log_loss ", log_loss(y_test, model2.predict_proba(X_test_scaled)))

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0,0]/(cm[0,1]+cm[0,0])
    print(cm)
    print("specificity ", specificity)

    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)

    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred)
    print("precision", precision)

    report = classification_report(y_test, y_pred, target_names=['what', 'what2'])
    print("report", report)

    recall = recall_score(y_test, y_pred)
    print("recall ", recall)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("f1", f1)

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, model2.predict(X_test_scaled))
    fpr, tpr, thresholds = roc_curve(y_test, model2.predict_proba(X_test_scaled)[:,1])
    area = np.trapz(tpr, fpr)
    print("area roc_auc", area)

    positive_count = 0
    negative_count = 0
    sentiments = model2.predict(X_totest_scaled)
    for sentiment in sentiments:
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
f_ranforest()