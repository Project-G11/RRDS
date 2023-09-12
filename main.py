from sklearn.model_selection import train_test_split
from baselines import MajorityClassModel, KeywordMatchingModel

# open data file
txt = open("dialog_acts.dat").readlines()
# split data in labels and instances list
labels = []
insts = []
for line in txt:
    idx = line.find(" ")
    labels.append(line[0:idx].lower())
    insts.append(line[idx+1:-1].lower())

# split data in test and training data
insts_train, insts_test, labels_train, labels_test = train_test_split(insts, labels, test_size=0.15, random_state=42)

# get baseline models
mc_model =  MajorityClassModel(labels_train)
km_model =  KeywordMatchingModel(labels_train)

# get accuracies
mc_acc = mc_model.evaluate(insts_test, labels_test)
print("Majority class model accuracy is", mc_acc)

km_acc = km_model.evaluate(insts_test, labels_test)
print("Keyword matching model accuracy is", km_acc)