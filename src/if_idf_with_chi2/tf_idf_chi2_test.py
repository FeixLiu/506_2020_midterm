from sklearn.metrics import mean_squared_error
import csv
import re
import numpy as np

path = '../data/train.csv'
stop_words = set({}.fromkeys([line.strip() for line in open('../data/stopword.txt')]))
data = []
label = []
ids = []
with open(path) as file:
    all_data = csv.reader(file)
    count = 0
    for i in all_data:
        if count == 0:
            count = 1
            continue
        count += 1
        # if count <= 100000:
        #     continue
        if i[5] != '':
            continue
        tmp = i[7] + ' ' + i[8]
        tmp = re.sub(r'[^a-zA-Z]', ' ', tmp)
        words = tmp.lower().split()
        data.append(' '.join(words))
        # label.append(float(i[5]))
        ids.append(i[0])
        # if count >= 105001:
        #     break

idf = {}
with open('../tf_idf_chi2/idf.txt') as file:
    for line in file:
        line = line.replace('\n', '')
        line = line.split('\t')
        idf[line[0]] = float(line[1])

tf = {}
keys = [1, 2, 3, 4, 5]
for key in keys:
    tmp = {}
    with open('../tf_idf_chi2/tf' + str(key) + '.txt') as file:
        for line in file:
            line = line.replace('\n', '')
            line = line.split('\t')
            tmp[line[0]] = float(line[1])
    tf[key] = tmp

pred = []
for line in data:
    line = line.split(' ')
    score = [0 for _ in range(5)]
    for word in line:
        try:
            idf_score = idf[word]
        except KeyError:
            continue
        for key in keys:
            try:
                tf_score = tf[key][word]
                score[key - 1] += tf_score * idf_score
            except KeyError:
                continue
    pred.append(float(np.argmax(score) + 1))
# print(mean_squared_error(label, pred))

with open('../tf_idf_chi2/rst.csv', 'w') as file:
    file.write('Id,Score' + '\n')
    for i in range(len(ids)):
        file.write(ids[i] + ',' + str(pred[i]) + '\n')
