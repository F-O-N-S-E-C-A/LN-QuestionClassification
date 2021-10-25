# LN 2021 - Tiago Fonseca - 102138 & João Lopes - 90732

import sys
import pandas as pd
import numpy as np
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


def main(validation_file, train_file):
    corpus = pd.read_csv(train_file, sep='\t', error_bad_lines=False, header=None)
    corpus_validation = pd.read_csv(validation_file, sep='\t', error_bad_lines=False, header=None)
    add_answers(corpus)
    add_answers(corpus_validation)
    print(corpus)
    tokenization(corpus)
    tokenization(corpus_validation)
    lemmatization(corpus)
    lemmatization(corpus_validation)
    #print(corpus)
    #Train_X, Test_X, train_Y, test_Y = model_selection.train_test_split(corpus['lemma'], corpus[0], test_size=0.3)

    print(corpus['lemma'][0])
    encoder = LabelEncoder()
    train_Y = encoder.fit_transform(corpus[0])
    test_Y = encoder.fit_transform(corpus_validation[0])

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(corpus['lemma'])
    train_X = Tfidf_vect.transform(corpus['lemma'])
    test_X = Tfidf_vect.transform(corpus_validation['lemma'])

    #print(Tfidf_vect.vocabulary_)
    #print(train_X_vector)
    support_vector_machine(train_X, test_X, train_Y, test_Y)

def add_answers(corpus):
    corpus[1] = corpus[1] + ' ' + corpus[2].astype(str)


def tokenization(corpus):
    corpus[1] = [entry.lower() for entry in corpus[1]]
    corpus[1] = [word_tokenize(entry) for entry in corpus[1]]


def lemmatization(corpus):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index, entry in enumerate(corpus[1]):
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
        corpus.loc[index, 'lemma'] = str(Final_words)


def support_vector_machine(Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y):
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight='balanced')
    #SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf, Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy: ", accuracy_score(predictions_SVM, Test_Y))

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