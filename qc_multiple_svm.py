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


def main(validation_file, train_file, operation):
    corpus = pd.read_csv(train_file, sep='\t', error_bad_lines=False, header=None, names=["Label", "Question", "Answer"])
    corpus_validation = pd.read_csv(validation_file, sep='\t', error_bad_lines=False, header=None, names=["Label", "Question", "Answer"])
    print(corpus)

    questionSVM, question_prediction = question_svm(corpus, corpus_validation)
    answerSVM, answer_prediction = answer_svm(corpus, corpus_validation)

    #print(question_prediction)
    #print(answer_prediction)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(corpus_validation['Label'])

    accuracy_question = accuracy_score(prob_vector_to_guess(question_prediction), labels)
    accuracy_answer = accuracy_score(prob_vector_to_guess(answer_prediction), labels)

    print('Accuracy Question SVM: ', accuracy_question, ' Accuracy Answer SVM: ', accuracy_answer)

    prediction = []
    for i in range(0, len(answer_prediction)):
        # max - Accuracy:  0.878
        # weighted_sum - Accuracy: 0.864
        if operation == 'max':
            if max(question_prediction[i]) > max(answer_prediction[i]):
                prediction.append(np.argmax(question_prediction[i]))
            else:
                prediction.append(np.argmax(answer_prediction[i]))
        elif operation == 'weighted_sum':
            # avr
            vec = []
            for j in range(0, len(question_prediction[i])):
                vec.append(question_prediction[i][j] * accuracy_question + answer_prediction[i][j] * accuracy_answer)
            prediction.append(vec.index(max(vec)))

    print('Accuracy: ', accuracy_score(prediction, labels))


def prob_vector_to_guess(list):
    res = []
    for l in list:
        res.append(np.argmax(l))
    return res


def question_svm(corpus, corpus_validation):
    tokenization(corpus, 'Question')
    tokenization(corpus_validation, 'Question')

    lemmatization(corpus, 'Question')
    lemmatization(corpus_validation, 'Question')

    # print(corpus)
    # Train_X, Test_X, train_Y, test_Y = model_selection.train_test_split(corpus['lemma'], corpus[0], test_size=0.3)

    encoder = LabelEncoder()
    train_Y = encoder.fit_transform(corpus['Label'])
    test_Y = encoder.fit_transform(corpus_validation['Label'])

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(corpus['lemma_Question'])
    train_X = Tfidf_vect.transform(corpus['lemma_Question'])
    test_X = Tfidf_vect.transform(corpus_validation['lemma_Question'])

    # print(Tfidf_vect.vocabulary_)
    # print(train_X_vector)
    svm = support_vector_machine(train_X, test_X, train_Y, test_Y)
    prediction = svm.predict_proba(test_X)
    return svm, prediction

def answer_svm(corpus, corpus_validation):
    #corpus['Answer'] = [str(entry) for entry in corpus['Answer']]
    tokenization(corpus, 'Answer')
    tokenization(corpus_validation, 'Answer')

    lemmatization(corpus, 'Answer')
    lemmatization(corpus_validation, 'Answer')

    # print(corpus)
    # Train_X, Test_X, train_Y, test_Y = model_selection.train_test_split(corpus['lemma'], corpus[0], test_size=0.3)

    encoder = LabelEncoder()
    train_Y = encoder.fit_transform(corpus['Label'])
    test_Y = encoder.fit_transform(corpus_validation['Label'])

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(corpus['lemma_Answer'])
    train_X = Tfidf_vect.transform(corpus['lemma_Answer'])
    test_X = Tfidf_vect.transform(corpus_validation['lemma_Answer'])

    # print(Tfidf_vect.vocabulary_)
    # print(train_X_vector)
    svm = support_vector_machine(train_X, test_X, train_Y, test_Y)
    prediction = svm.predict_proba(test_X)
    return svm, prediction

def add_answers(corpus):
    corpus[1] = corpus[1] + ' ' + corpus[2].astype(str)


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
    print(corpus)


def support_vector_machine(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y):
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight='balanced', probability=True)
    #SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
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
            main(validation_file=sys.argv[2], train_file=sys.argv[4], operation='weighted_sum')
        else:
            man()
    else:
        man()