from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# model that assigns labels to data using logistic regression
class LogisticRegressionModel:

    # initialize model with training labels
    def __init__(self, insts_train, insts_test, labels_train, labels_test):
        # vectorize data using bag of words model
        vectorizer = CountVectorizer()
        vectorizer.fit(insts_train)
        input_train = vectorizer.transform(insts_train)
        input_test  = vectorizer.transform(insts_test)
        self.build(input_train, input_test, labels_train, labels_test)

    # build the model
    def build(self, input_train, input_test, labels_train, labels_test):
        classifier = LogisticRegression(max_iter=500)
        classifier.fit(input_train, labels_train)
        self.evaluate(classifier, input_test, labels_test)
        
    # evaluates model and returns the accuracy
    def evaluate(self, classifier, input_test, labels_test):
        print("Logistic regression accuracy is", classifier.score(input_test, labels_test))