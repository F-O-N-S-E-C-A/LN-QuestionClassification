import pandas as pd

corpus = pd.read_csv('trainWithoutDev.txt', sep='\t', error_bad_lines=False, header=None)
corpus_validation = pd.read_csv('dev.txt', sep='\t', error_bad_lines=False, header=None)

classes_corpus = {}
for c in corpus[0]:
    if c in classes_corpus:
        classes_corpus[c] += 1
    else:
        classes_corpus[c] = 0

print(classes_corpus)

classes_validation = {}
for c in corpus_validation[0]:
    if c in classes_validation:
        classes_validation[c] += 1
    else:
        classes_validation[c] = 0

print(classes_validation)