import csv
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
import numpy as np

path = '../data/train.csv'
stop_words = set({}.fromkeys([line.strip() for line in open('../data/stopword.txt')]))

all_file = []
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
        tmp = i[7]
        tmp = re.sub(r'[^a-zA-Z]', ' ', tmp)
        words = tmp.lower().split()
        words = [w for w in words if w not in stop_words]
        all_file.append(' '.join(words))
        label.append(float(i[5]))
        count += 1
        if count == 100001:
            break

vectorizer = CountVectorizer(decode_error='replace', max_features=10000)
vectorizer.fit(all_file)
vocab = {}.fromkeys(vectorizer.get_feature_names())
idf = {}
for line in all_file:
    line = line.split(' ')
    line = set(line)
    for word in line:
        try:
            a = vocab[word]
            try:
                idf[word] += 1
            except KeyError:
                idf[word] = 1
        except KeyError:
            continue
total_file = len(all_file)
for i in idf.keys():
    idf[i] = math.log(total_file / idf[i])

vocab_dict = {}
dict_vocab = {}
idf_items = sorted(idf.items(), key=lambda x: x[1])
for i in range(len(idf_items)):
    vocab_dict[idf_items[i][0]] = i
    dict_vocab[i] = idf_items[i][0]
idf_vec = np.zeros((10000, ))
for i in dict_vocab.keys():
    idf_vec[i] = idf[dict_vocab[i]]

vec = CountVectorizer(decode_error='replace', vocabulary=vocab.keys())
nb = MultinomialNB()
count = 0
while count < len(all_file):
    print('train' + " " + str(count))
    if count + 1000 < len(all_file):
        line = all_file[count: count + 1000]
        lb = label[count: count + 1000]
    else:
        line = all_file[count: len(all_file)]
        lb = label[count: len(all_file)]
    count += 1000
    line_code = []
    for sentence in line:
        tmp = np.zeros((10000, ))
        words = sentence.split(' ')
        wc = 0
        for word in words:
            wc += 1
            try:
                tmp[vocab_dict[word]] += 1
            except KeyError:
                continue
        tmp = np.divide(tmp, wc)
        tmp = np.multiply(tmp, idf_vec)
        line_code.append(tmp.tolist())
    nb.partial_fit(line_code, lb, classes=np.unique(lb))

test = []
test_label = []
ids = []
with open(path) as file:
    all_data = csv.reader(file)
    count = 0
    for i in all_data:
        if count == 0:
            count = 1
            continue
        count += 1
        if count <= 100000:
            continue
        if i[5] == '':
            continue
        tmp = i[7] + " " + i[8]
        tmp = re.sub(r'[^a-zA-Z]', ' ', tmp)
        words = tmp.lower().split()
        test.append(' '.join(words))
        test_label.append(float(i[5]))
        ids.append(i[0])
        if count >= 105001:
            break
pred = []
count = 0
while count < len(test):
    print('test' + " " + str(count))
    if count + 1000 < len(test):
        line = all_file[count: count + 1000]
        lb = label[count: count + 1000]
    else:
        line = all_file[count: len(test)]
        lb = label[count: len(test)]
    count += 1000
    line_code = []
    for sentence in line:
        tmp = np.zeros((10000,))
        words = sentence.split(' ')
        wc = 0
        for word in words:
            wc += 1
            try:
                tmp[vocab_dict[word]] += 1
            except KeyError:
                continue
        tmp = np.divide(tmp, wc)
        tmp = np.multiply(tmp, idf_vec)
        line_code.append(tmp.tolist())
    temp = nb.predict(line_code)
    for i in temp:
        pred.append(i)
print(mean_squared_error(test_label, pred))
