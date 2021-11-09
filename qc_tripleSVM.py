# LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732

import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import random
from plot import PLOT

random.seed(10)
np.random.seed(999)

MAX = 'max'
WEIGHTED_SUM = 'weighted_sum'
JOINT_PROB = 'joint_probability'


def main(validation_file, train_file, operation):
    corpus = pd.read_csv(train_file, sep='\t', error_bad_lines=False, header=None, names=["Label", "Question", "Answer"])
    corpus_validation = pd.read_csv(validation_file, sep='\t', error_bad_lines=False, header=None, names=["Label", "Question", "Answer"])
    print(corpus)

    concat_questions_answers(corpus)
    concat_questions_answers(corpus_validation)

    questionSVM, question_prediction = build_svm(corpus, corpus_validation, data_part='Question')
    answerSVM, answer_prediction = build_svm(corpus, corpus_validation, data_part='Answer')
    q_and_a_SVM, q_and_a_prediction = build_svm(corpus, corpus_validation, data_part='Q+A')

    encoder = LabelEncoder()
    labels = encoder.fit_transform(corpus_validation['Label'])
    prediction = []

    # max - Accuracy:  0.878
    # weighted_sum - Accuracy: 0.864
    # joint_probability - Accuracy: 0.864

    if operation == 'max':
        for i in range(0, len(answer_prediction)):
            if max(question_prediction[i]) > max(answer_prediction[i]) and max(question_prediction[i]) > max(q_and_a_prediction[i]):
                prediction.append(np.argmax(question_prediction[i]))
            elif max(q_and_a_prediction[i]) > max(question_prediction[i]) and max(q_and_a_prediction[i]) > max(answer_prediction[i]):
                prediction.append(np.argmax(q_and_a_prediction[i]))
            else:
                prediction.append(np.argmax(answer_prediction[i]))

    elif operation == 'weighted_sum':

        accuracy_question = accuracy_score(prob_vector_to_guess(question_prediction), labels)
        accuracy_answer = accuracy_score(prob_vector_to_guess(answer_prediction), labels)
        accuracy_q_and_a = accuracy_score(prob_vector_to_guess(q_and_a_prediction), labels)

        print('Accuracy Question SVM: ', accuracy_question, ' Accuracy Answer SVM: ', accuracy_answer, ' Accuracy Q+A SVM: ', accuracy_q_and_a)

        for i in range(0, len(answer_prediction)):
            vec = []
            for j in range(0, len(question_prediction[i])):
                vec.append(question_prediction[i][j] * accuracy_question + answer_prediction[i][j] * accuracy_answer + q_and_a_prediction[i][j] * accuracy_q_and_a)
            prediction.append(vec.index(max(vec)))

    elif operation == 'joint_probability':
        for i in range(0, len(answer_prediction)):
            vec = []
            for j in range(0, len(question_prediction[i])):
                vec.append(question_prediction[i][j] * answer_prediction[i][j] * q_and_a_prediction[i][j])
            prediction.append(vec.index(max(vec)))

    print('Accuracy: ', accuracy_score(prediction, labels))

    '''for i in range(len(prediction)):
        if (prediction[i] != labels[i]):
            print(corpus['Q+A'][i])'''


def concat_questions_answers(corpus):
    corpus['Q+A'] = corpus['Question'] + ' ' + corpus['Answer'].astype(str)


def prob_vector_to_guess(list):
    res = []
    for l in list:
        res.append(np.argmax(l))
    return res


def build_svm(corpus, corpus_validation, data_part):
    tokenization(corpus, data_part)
    tokenization(corpus_validation, data_part)

    word_lemmatizer(corpus, data_part)
    word_lemmatizer(corpus_validation, data_part)

    print(corpus)

    encoder = LabelEncoder()
    train_Y = encoder.fit_transform(corpus['Label'])
    test_Y = encoder.fit_transform(corpus_validation['Label'])

    Tfidf_vect = TfidfVectorizer(max_features=70000) #10000
    lemma = 'lemma_' + data_part
    Tfidf_vect.fit(corpus[lemma])
    train_X = Tfidf_vect.transform(corpus[lemma])
    test_X = Tfidf_vect.transform(corpus_validation[lemma])

    # print(Tfidf_vect.vocabulary_)
    # print(train_X_vector)

    #PLOT(train_X)
    #PLOT(test_X)

    svm = support_vector_machine(train_X, train_Y)
    prediction = svm.predict_proba(test_X)
    return svm, prediction


def tokenization(df, index):
    df[index] = [str(entry).lower() for entry in df[index]]
    df[index] = [word_tokenize(entry) for entry in df[index]]


def word_lemmatizer(corpus, column):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(corpus[column]):
        Final_words = []

        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and (word.isalpha() or any(c.isdigit() for c in word)):
                Final_words.append(WordNetLemmatizer().lemmatize(word, tag_map[tag[0]]))
                Final_words.append(word)
        corpus.loc[index, 'lemma_' + column] = str(Final_words)


def support_vector_machine(Train_X_Tfidf, Train_Y):
    SVM = svm.SVC(C=1.0, kernel='linear' , class_weight='balanced', probability=True)
    SVM.fit(Train_X_Tfidf, Train_Y)
    return SVM


def man():
    print('LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732\n')
    print('How to run:')
    print('python qc.py –test <NAMEOFTESTFILE> –train <NAMEOFTHETRAINFILE> > results.txt\n')
    sys.exit()


# run command: python3 qc.py -test dev.txt -train trainWithoutDev.txt
if __name__ == '__main__':
    if len(sys.argv) == 5:
        if sys.argv[1] == '-test' and sys.argv[3] == '-train':
            main(validation_file=sys.argv[2], train_file=sys.argv[4], operation=MAX)
        else:
            man()
    else:
        man()
