import numpy
import os
import csv
import pandas as pd
import string
import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics



# 0:cyberbullying 1:normal
def extract_feature():

    embedding = {}  # key:userId value:embeding
    with open('user.emd', mode='r') as embeds:
        for line in embeds.readlines():
            data = line.strip().split()
            userId = data[0]
            embed = [float(i) for i in data[1:]]
            embedding[userId] = numpy.array(embed)

    user2label = {}  # key:userId, value:user label
    all_data = []
    all_label = []
    with open('user_label.csv', encoding="utf8", errors='ignore', mode='r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            user2label[row[0]] = row[1]
            all_data.append(row[0])
            all_label.append(row[1])
    posData = []
    negData = []
    poslabel = []
    neglabel = []
    for userId in user2label.keys():
        embed = embedding.get(userId, numpy.zeros(128)).tolist()
        label = user2label[userId]
        # print(label)
        if label == '1':
            posData.append(embed)
            poslabel.append(1)
            # print('pos')
        else:
            negData.append(embed)
            neglabel.append(0)
            # print('neg')
    return posData, negData, poslabel, neglabel


def prepare_data(pos, neg):
    label = []
    data = []
    for p in pos:
        label.append(1)
        data.append(p)
    for n in neg:
        label.append(0)
        data.append(n)
    return data, label


def main():
    posData, negData, poslabel, neglabel = extract_feature()
    # print(len(posData))
    pos_train, pos_test = train_test_split(posData, test_size=0.2, random_state=0)
    neg_train, neg_test = train_test_split(negData, test_size=0.2, random_state=0)
    pos_label_train,pos_label_test = train_test_split(poslabel, test_size=0.2, random_state=0)
    neg_label_train, neg_label_test = train_test_split(neglabel, test_size=0.2, random_state=0)
    train_data = pos_train + neg_train
    test_data = pos_test + neg_test
    train_label = pos_label_train + neg_label_train
    test_label = pos_label_test + neg_label_test
    # train_data, train_label = prepare_data(pos_train,neg_train)
    # test_data, test_label = prepare_data(pos_test,neg_test)
    tr_features = numpy.array(train_data)
    tr_features_test = numpy.array(test_data)

    # print(train_data)

    # svm
    print('svm starts')
    clf = svm.SVC(kernel='linear', probability=True)
    # clf = svm.SVC(C = 10, gamma = 0.001, kernel='rbf')
    clf = clf.fit(tr_features, train_label)
    output_prediction_train = clf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\nSVM:\n', classification_report(test_label, clf.predict(tr_features_test)))

    # random forest
    clf1 = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1, max_depth=2)
    clf1 = clf1.fit(tr_features, train_label)
    output_prediction_train = clf1.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    # print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\nRandom Forest:\n', classification_report(test_label, clf1.predict(tr_features_test)))

    # logic regression
    regr = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200)
    regr = regr.fit(tr_features, train_label)
    output_prediction_train = regr.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    # print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\nLogic Regression:\n', classification_report(test_label, regr.predict(tr_features_test)))

    # ada boosting
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    classifier = classifier.fit(tr_features, train_label)
    output_prediction_train = classifier.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    # print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\n Ada Boosting:\n', classification_report(test_label, classifier.predict(tr_features_test)))

    # SDG classfier
    sdgclf = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(max_iter=1000, tol=1e-3))
    sdgclf.fit(tr_features, train_label)
    output_prediction_train = sdgclf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\n SDG classifier:\n', classification_report(test_label, sdgclf.predict(tr_features_test)))

    # Gaussian naive bayes
    nbclf = GaussianNB()
    nbclf.fit(tr_features, train_label)
    output_prediction_train = nbclf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\n Naive Bayes:\n', classification_report(test_label, nbclf.predict(tr_features_test)))

    fprsvm, tprsvm, thresholdssvm = metrics.roc_curve(test_label, clf.predict_proba(tr_features_test)[:, 1])
    fprrf, tprrf, thresholdsrf = metrics.roc_curve(test_label, clf1.predict_proba(tr_features_test)[:, 1])
    fprlr, tprlr, thresholdslr = metrics.roc_curve(test_label, regr.predict_proba(tr_features_test)[:, 1])
    fprab, tprab, thresholdsab = metrics.roc_curve(test_label, classifier.predict_proba(tr_features_test)[:, 1])
    fprnb, tprnb, thresholdsnb = metrics.roc_curve(test_label, nbclf.predict_proba(tr_features_test)[:, 1])
    # fprsdg, tprsdg, thresholdssdg = metrics.roc_curve(test_label, sdgclf.predict_proba(tr_features_test)[:, 1])

    plt.figure(figsize=(15, 8))
    lw = 2
    plt.plot(fprrf, tprrf, color='darkorange',
             lw=lw, label='ROC curve  Random Forest (area = %0.2f)' % auc(fprrf, tprrf))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(15, 8))
    lw = 2
    plt.plot(fprsvm, tprsvm, color='darkorange',
             lw=lw, label='ROC curve  SVM (area = %0.2f)' % auc(fprsvm, tprsvm))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(15, 8))
    lw = 2
    plt.plot(fprlr, tprlr, color='darkorange',
             lw=lw, label='ROC curve  Logic Regression (area = %0.2f)' % auc(fprlr, tprlr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(15, 8))
    lw = 2
    plt.plot(fprab, tprab, color='darkorange',
             lw=lw, label='ROC curve  Ada Boosting (area = %0.2f)' % auc(fprab, tprab))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(15, 8))
    lw = 2
    plt.plot(fprnb, tprnb, color='darkorange',
             lw=lw, label='ROC curve  Naive Bay (area = %0.2f)' % auc(fprnb, tprnb))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
