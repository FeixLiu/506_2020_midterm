import csv
import re
import math
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

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

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()
for key in data.keys():
    for i in data[key]:
        i = i.split(' ')
        for word in i:
            try:
                word_fd[word] += 1
            except KeyError:
                word_fd[word] = 1
            try:
                label_word_fd[key][word] += 1
            except KeyError:
                label_word_fd[key][word] = 1

keys = [1, 2, 3, 4, 5]
count = [0 for _ in range(6)]
total_count = 0
for key in keys:
    count[key] = label_word_fd[key].N()
    total_count += label_word_fd[key].N()
word_score = {}
for word in word_fd.keys():
    score = 0
    for key in keys:
        score += BigramAssocMeasures.chi_sq(label_word_fd[key][word], (word_fd[word], count[key]), total_count)
    word_score[word] = score
best = sorted(word_score.items(), key=lambda x: x[1], reverse=True)

vocab = {}
for i in range(10000):
    vocab[best[i][0]] = i

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
with open('../tf_idf_chi2/idf.txt', 'w') as file:
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
    with open('../tf_idf_chi2/tf' + str(key) + '.txt', 'w') as file:
        for i in tf:
            file.write(i[0] + "\t" + str(i[1]) + '\n')
