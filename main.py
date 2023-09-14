from sklearn.model_selection import train_test_split
from baselines import MajorityClassModel, KeywordMatchingModel
from nn_classifier import NNClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# If True the duplicates are removed from the data
no_duplicates = True

# open data file
txt = open("dialog_acts.dat").readlines()
# split data in labels and instances list
labels = []
insts = []
lines = []
for line in txt:
    if no_duplicates:
        lines.append(line)
    else:
        idx = line.find(" ")
        labels.append(line[0:idx].lower())
        insts.append(line[idx+1:-1].lower())

if no_duplicates:
    # remove duplicates from data
    temp = set()
    result = [i for i in lines if not(i in temp or temp.add(i))]
    labels = []
    insts = []
    for line in result:
        idx = line.find(" ")
        labels.append(line[0:idx].lower())
        insts.append(line[idx+1:-1].lower())

    
# split data in test and training data
insts_train, insts_test, labels_train, labels_test = train_test_split(insts, labels, test_size=0.15, random_state=42)

# get baseline models
train_labels = labels_train
test_labels = labels_test
test_insts = insts_test

mc_model =  MajorityClassModel(train_labels)
km_model =  KeywordMatchingModel(train_labels)

# get accuracies
mc_acc = mc_model.evaluate(test_insts, test_labels)
print("Majority class model accuracy is", mc_acc)

km_acc = km_model.evaluate(test_insts, test_labels)
print("Keyword matching model accuracy is", km_acc)

# vectorize data using bag of words model
vectorizer = CountVectorizer()
vectorizer.fit(insts_train)
input_train = vectorizer.transform(insts_train)
input_test  = vectorizer.transform(insts_test)

# logistic regression classifier
classifier = LogisticRegression()
classifier.fit(input_train, labels_train)
score = classifier.score(input_test, labels_test)
print("Logistic regression accuracy is", score)

# Create, train and evaluate the FFNN Classifier
nn_model = NNClassifier(insts_train, insts_test, labels_train, labels_test)