# LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import random

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

    #print(question_prediction)
    #print(answer_prediction)

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

    #lemmatization(corpus, data_part)
    #lemmatization(corpus_validation, data_part)
    pre_processing(corpus, data_part)
    pre_processing(corpus_validation, data_part)



    print(corpus)
    # Train_X, Test_X, train_Y, test_Y = model_selection.train_test_split(corpus['lemma'], corpus[0], test_size=0.3)

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

    svm = support_vector_machine(train_X, test_X, train_Y, test_Y)
    prediction = svm.predict_proba(test_X)
    return svm, prediction


def tokenization(df, index):
    df[index] = [str(entry).lower() for entry in df[index]]
    df[index] = [word_tokenize(entry) for entry in df[index]]


def lemmatization(corpus, column):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(corpus[column]):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        corpus.loc[index, 'lemma_'+column] = str(Final_words)
    #print(corpus)


def pre_processing(corpus, column):
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
    # print(corpus)

    '''for index, entry in enumerate(corpus[column]):
        Final_words = []
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and (word.isalpha() or any(c.isdigit() for c in word)):
                word_Final = word
                Final_words.append(word_Final)
        corpus.loc[index, 'lemma_' + column] = str(Final_words)
    # print(corpus)'''


def support_vector_machine(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y):
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight='balanced', probability=True)
    SVM.fit(Train_X_Tfidf, Train_Y)
    # predict the labels on validation dataset
    #predictions_SVM = SVM.predict_proba(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    #print("SVM Accuracy: ", accuracy_score(predictions_SVM, Test_Y))
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