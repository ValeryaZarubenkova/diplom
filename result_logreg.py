import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import sklearn.model_selection
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss, recall_score
import pickle
import result_output

def f_logreg():
    products = pd.read_csv('result_learyted.csv', sep=',')

    # Split data into training and testing sets
    Xx = products[['word_vectors']]
    y = products['ozenka']

    print(len(Xx), len(y))


    X_train, X_test, y_train, y_test = train_test_split(Xx, y, test_size=0.25)
    # Convert string vectors to numpy arrays
    X_train = np.array([ast.literal_eval(x) for x in X_train['word_vectors']])
    X_test = np.array([ast.literal_eval(x) for x in X_test['word_vectors']])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('scaler', X_train_scaled, X_train_scaled)
    import time 
    t0 = time.time()

    # Define logistic regression model
    model = sklearn.linear_model.LogisticRegression(C=1.0, max_iter=1000)

    # Train model on training data
    model = model.fit(X_train_scaled, y_train)

    #pkl_file = "pickle_model.pkl"
    #save model
    #with open(pkl_file, 'wb') as file:
     #   pickle.dump(model, file)
    #import model
    #with open(pkl_file, 'rb') as file:
    #    model = pickle.load(file)   


    y_pred = model.predict(X_test_scaled)

    t1 = time.time() - t0
    print("TIME = ", t1)
        # Evaluate model on testing data
    accuracy = model.score(X_test_scaled, y_test)
    print('Accuracy:', accuracy)

    #from sklearn.metrics import f1_score
    #thresholds = [ 0.508, 0.509, 0.51, 0.515, 0.55]
    #for threshold in thresholds:
     #   y_pred_threshold = [1 if proba >= threshold else 0 for proba in model.predict_proba(X_test_scaled)[:, 1]]
      #  f1 = f1_score(y_test, y_pred_threshold)
       # print(f'Threshold: {threshold}, F1-score: {f1}')

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0,0]/(cm[0,1]+cm[0,0])
    print(cm)
    print("specificity ", specificity)

    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("f1", f1)

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test_scaled))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
    area = np.trapz(tpr, fpr)
    print("area roc_auc", area)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()



    train_predictions = model.predict(X_train_scaled)
    train_roc_auc = roc_auc_score(y_train, train_predictions)
    test_predictions = model.predict(X_test_scaled)
    test_roc_auc = roc_auc_score(y_test, test_predictions)
    overfit_auc = train_roc_auc - test_roc_auc
    print("overfit", overfit_auc)

    from sklearn.metrics import precision_score
    precision = precision_score(y_test, y_pred)
    print("precision", precision)

    report = classification_report(y_test, y_pred, target_names=['what', 'what2'])
    print("report", report)
    print("log_loss ", log_loss(y_test, model.predict_proba(X_test_scaled)))

    recall = recall_score(y_test, y_pred)
    print("recall ", recall)
    
    # Загрузка тестового набора данных
    test_data = pd.read_csv('some_learyted.csv', sep=',')
    X_totest = test_data[['word_vectors']]
    X_totest = np.array([ast.literal_eval(x) for x in X_totest['word_vectors']])

    X_totest_scaled = scaler.transform(X_totest)

    positive_count = 0
    negative_count = 0

    # Определение тональности комментариев с помощью модели
    sentiments = model.predict_proba(X_totest_scaled)
    sentiments1 = model.predict(X_totest_scaled)
    test_data['predicted'] = sentiments1
    #print(test_data)
    arr = [0]*len(sentiments)
    i = -1
    # Подсчет количества положительных и отрицательных комментариев
    for sentiment in sentiments:
        i = i+1
        if sentiment[1] > 0.5:
            positive_count += 1
        elif sentiment[1] <= 0.5:
            negative_count += 1   
        arr[i] = sentiment[1]

    test_data['predicted[1]'] = arr
    test_data.to_csv('itog.csv', index=False)


    # Вычисление процента положительных и отрицательных комментариев
    total_count = len(test_data)

    res = result_output.f_output(total_count, positive_count, negative_count)
    return res
f_logreg()