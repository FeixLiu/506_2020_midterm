import csv
import re
import math

from sklearn.feature_extraction.text import CountVectorizer

path = '../data/train.csv'
stop_words = set({}.fromkeys([line.strip() for line in open('../data/stopword.txt')]))
data = {}
all_file = []
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
        try:
            data[int(float(i[5]))].append(' '.join(words))
        except KeyError:
            data[int(float(i[5]))] = [' '.join(words)]
        all_file.append(' '.join(words))
        count += 1

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
idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)
with open('../tf_idf/idf.txt', 'w') as file:
    for i in idf:
        file.write(i[0] + "\t" + str(i[1]) + '\n')

for key in data.keys():
    count = 0
    tf = {}
    for line in data[key]:
        line = line.split(' ')
        count += len(line)
        for word in line:
            try:
                a = vocab[word]
                try:
                    tf[word] += 1
                except KeyError:
                    tf[word] = 1
            except KeyError:
                continue
    for i in tf.keys():
        tf[i] = tf[i] / count
    tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    with open('../tf_idf/tf' + str(key) + '.txt', 'w') as file:
        for i in tf:
            file.write(i[0] + "\t" + str(i[1]) + '\n')
