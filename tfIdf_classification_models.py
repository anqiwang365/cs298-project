import pandas as pd
import string
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import csv

def load_data():

    posData = []
    negData = []
    poslabel = []
    neglabel = []
    with open('tweet_dataset_edited.csv', encoding="utf8", errors='ignore', mode='r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            content = preprocess_data(row[1])
            label = row[4]  # 0:cyber; 1:normal
            if label == '1':
                posData.append(content)
                poslabel.append(1)
                # print('pos')
            else:
                negData.append(content)
                neglabel.append(0)
                # print('neg')
    return posData, negData, poslabel, neglabel

def preprocess_data(data):
    tokens = []
    # 1. remove url
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
    # 2. remove punctuation
    data = data.translate(str.maketrans('', '', string.punctuation))
    # 3. remove emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)
    # 4. remove '#mkr
    data = data.replace('mkr', '')
    # 5. replace multiple space with single space
    data = re.sub(r'\s+', ' ', data)
    # 6. convert to lower case
    data = data.lower()
    # 7. remove stop words
    # stop_words = set(open('stopwords.txt', 'r').read().split())
    stop_words = stopwords.words('english')
    data = nltk.word_tokenize(data)
    data = [word for word in data if word not in stop_words]
    # 8. lemmatizer
    lemmatizer = WordNetLemmatizer()
    for word in data:
        lword = lemmatizer.lemmatize(word)
        lword = str(PorterStemmer().stem(lword))
        tokens.append(lword)
        # print(lword)
    return ' '.join(tokens)


def prepare_data(pos, neg):
    posData = []
    negData = []
    posLabel = []
    negLabel = []
    for p in pos['tweet_content']:
        posData.append(preprocess_data(p))
        posLabel.append(0)
    for n in neg['tweet_content']:
        negData.append(preprocess_data(n))
        negLabel.append(1)
    data = posData + negData
    # label = np.concatenate((pos['category'].values, neg['category'].values))
    label = posLabel + negLabel
    return data, label


def tokenize(data):
    t = []
    for line in data:
        l = nltk.word_tokenize(line)
        for w in l:
            t.append(w)
    return t


def main():

    # split positive and negative dataset
    posData, negData, poslabel, neglabel = load_data()
    pos_train, pos_test = train_test_split(posData, test_size=0.2, random_state=0)
    neg_train, neg_test = train_test_split(negData, test_size=0.2, random_state=0)
    pos_label_train, pos_label_test = train_test_split(poslabel, test_size=0.2, random_state=0)
    neg_label_train, neg_label_test = train_test_split(neglabel, test_size=0.2, random_state=0)
    train_data = pos_train + neg_train
    test_data = pos_test + neg_test
    train_label = pos_label_train + neg_label_train
    test_label = pos_label_test + neg_label_test

    vec = TfidfVectorizer(binary=True, use_idf=True)
    tr_features = vec.fit_transform(train_data)
    tr_features_test = vec.transform(test_data)
    print('feature amount')
    print(tr_features.shape[0])

    # svm
    # clf = svm.SVC(kernel='linear', probability=True)
    clf = svm.SVC(C = 10, gamma = 0.001, kernel='rbf',probability=True)
    clf = clf.fit(tr_features, train_label)
    output_prediction_train = clf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\nSVM:\n', classification_report(test_label, clf.predict(tr_features_test)))

    # random forest
    clf1 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1, max_depth=5)
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

    # Multinomial naive bayes
    nbclf = MultinomialNB()
    nbclf.fit(tr_features, train_label)
    output_prediction_train = nbclf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\n Naive Bayes:\n', classification_report(test_label, nbclf.predict(tr_features_test)))

    # SDG classfier
    sdgclf = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(max_iter=1000, tol=1e-3))
    sdgclf.fit(tr_features, train_label)
    output_prediction_train = sdgclf.predict(tr_features_test)
    output_train_accuracy = metrics.accuracy_score(output_prediction_train, test_label)
    print("Accuracy on the Training dataset.", output_train_accuracy)
    print('\n SDG classifier:\n', classification_report(test_label, sdgclf.predict(tr_features_test)))

    fprsvm, tprsvm, thresholdssvm = metrics.roc_curve(test_label, clf.predict_proba(tr_features_test)[:, 1])
    fprrf, tprrf, thresholdsrf = metrics.roc_curve(test_label, clf1.predict_proba(tr_features_test)[:, 1])
    fprlr, tprlr, thresholdslr = metrics.roc_curve(test_label, regr.predict_proba(tr_features_test)[:, 1])
    fprab, tprab, thresholdsab = metrics.roc_curve(test_label, classifier.predict_proba(tr_features_test)[:, 1])
    fprnb, tprnb, thresholdsnb = metrics.roc_curve(test_label, nbclf.predict_proba(tr_features_test)[:, 1])
    # # fprsdg, tprsdg, thresholdssdg = metrics.roc_curve(test_label, sdgclf.predict_proba(tr_features_test)[:, 1])

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
