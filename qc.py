# LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732

import sys
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import random

import stats

random.seed(10)
np.random.seed(999)


def main(validation_file, train_file):
    corpus = pd.read_csv(train_file, sep='\t', error_bad_lines=False, header=None)
    corpus_validation = pd.read_csv(validation_file, sep='\t', error_bad_lines=False, header=None)
    add_answers(corpus)
    add_answers(corpus_validation)

    tokenization(corpus)
    tokenization(corpus_validation)

    pre_p(corpus)
    pre_p(corpus_validation)

    encoder = LabelEncoder()
    train_Y = encoder.fit_transform(corpus[0])
    test_Y = encoder.fit_transform(corpus_validation[0])

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(corpus['lemma'])
    train_X = Tfidf_vect.transform(corpus['lemma'])
    #print(train_X)
    test_X = Tfidf_vect.transform(corpus_validation['lemma'])

    support_vector_machine(train_X, test_X, train_Y, test_Y, corpus_validation)

    #PLOT(corpus, train_X)
    #PLOT(corpus_validation, test_X)


def add_answers(corpus):
    corpus[1] = corpus[1] + ' ' + corpus[2].astype(str)


def tokenization(corpus):
    corpus[1] = [entry.lower() for entry in corpus[1]]
    corpus[1] = [word_tokenize(entry) for entry in corpus[1]]


def pre_p(corpus):
    stop_words = ['what', 'who', 'when', 'which', 'this', 'in', 'on', 'at', 'the', 'if', 'a', 'it', 'he', 'she', 'they', 'we']
    for index, entry in enumerate(corpus[1]):
        Final_words = []
        for word in entry:

            if word not in stop_words and word.isalpha():
                Final_words.append(word)

        corpus.loc[index, 'lemma'] = str(Final_words)


def support_vector_machine(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y, corpus_validation):
    SVM = svm.SVC(kernel='linear')
    SVM.fit(Train_X_Tfidf, Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    print("SVM Accuracy: ", accuracy_score(predictions_SVM, Test_Y))

    '''accuracies = []
    occurencies = {'LITERATURE': 0, 'HISTORY': 0, 'SCIENCE': 0, 'MUSIC': 0, 'GEOGRAPHY': 0}
    for i in range(len(predictions_SVM)):
        if (predictions_SVM[i] == Test_Y[i]):
            occurencies[corpus_validation[0][i]] += 1
    classes_validation = {}
    for c in corpus_validation[0]:
        if c in classes_validation:
            classes_validation[c] += 1
        else:
            classes_validation[c] = 1
    for keys in occurencies.keys():
        accuracies.append(occurencies[keys] / classes_validation[keys])

    print(accuracies)'''


def man():
    print('LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732\n')
    print('How to run:')
    print('python qc.py –test <NAMEOFTESTFILE> –train <NAMEOFTHETRAINFILE> > results.txt\n')
    sys.exit()


# run command: python3 qc.py -test dev.txt -train trainWithoutDev.txt
if __name__ == '__main__':
    if len(sys.argv) == 5:
        if sys.argv[1] == '-test' and sys.argv[3] == '-train':
            main(validation_file=sys.argv[2], train_file=sys.argv[4])
        else:
            man()
    else:
        man()
