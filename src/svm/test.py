import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
import csv
import re


def read_data(path):
    data = []
    count = 0
    label = []
    target = []
    with open(path) as file:
        all_data = csv.reader(file)
        for i in all_data:
            if count == 0:
                count = 1
                continue
            count += 1
            # if count <= 100000:
            #     continue
            if i[5] != '':
                continue
            data.append(i[7] + " " + i[8])
            # label.append(float(i[5]))
            target.append(i[0])
            # if count > 105001:
            #     break
    return data, label, target


def clean_data(data):
    stop_words = set({}.fromkeys([line.strip() for line in open('../data/stopword.txt')]))
    for i in range(len(data)):
        text = data[i]
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        words = [w for w in words if w not in stop_words]
        data[i] = ' '.join(words)
    return data


if __name__ == '__main__':
    data, label, target = read_data('../data/train.csv')
    data = clean_data(data)
    vec = CountVectorizer(decode_error='replace', vocabulary=pickle.load(open('../bag_of_word/feature.pkl', 'rb')))
    model = joblib.load('../bag_of_word/model.pkl')
    pred = []
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
        temp = model.predict(line_code)
        for i in temp:
            pred.append(i)
    # print(mean_squared_error(label, pred))
    with open('../bag_of_word/rst.csv', 'w') as file:
        file.write('Id,Score\n')
        for i in range(len(target)):
            file.write(target[i] + ',' + str(float(pred[i])) + "\n")