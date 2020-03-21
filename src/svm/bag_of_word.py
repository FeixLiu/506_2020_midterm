import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import joblib
import numpy as np


def read_data(path):
    data = []
    label = []
    with open(path) as file:
        all_data = csv.reader(file)
        count = 0
        for i in all_data:
            if count == 0:
                count = 1
                continue
            if i[5] == '':
                continue
            data.append(i[7])
            label.append(int(float(i[5])))
            count += 1
            # if count == 100001:
            #     break
    return data, label


def clean_data(data):
    stop_words = set({}.fromkeys([line.strip() for line in open('../data/stopword.txt')]))
    for i in range(len(data)):
        text = data[i]
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        words = [w for w in words if w not in stop_words]
        data[i] = ' '.join(words)
    return data


def bag_of_words(data):
    vectorizer = CountVectorizer(decode_error='replace', max_features=10000)
    vectorizer.fit(data)
    with open('../bag_of_word/feature.pkl', 'wb') as file:
        pickle.dump(vectorizer.vocabulary_, file)
    return vectorizer.get_feature_names()


if __name__ == '__main__':
    data, label = read_data('../data/train.csv')
    data = clean_data(data)
    vocab = bag_of_words(data)
    lr = SGDClassifier(max_iter=10000)
    vec = CountVectorizer(decode_error='replace', vocabulary=vocab)
    count = 0
    while count < len(data):
        print(str(count) + " " + str(len(data)))
        if count + 1000 < len(data):
            line = data[count: count + 1000]
            lb = label[count: count + 1000]
        else:
            line = data[count: len(data)]
            lb = label[count: len(data)]
        count += 1000
        line_code = vec.fit_transform(line).toarray()
        lr.partial_fit(line_code, lb, classes=np.unique(lb))
    joblib.dump(lr, '../bag_of_word/model.pkl')
