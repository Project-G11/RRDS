# model that assigns labels to data based on the majority class of the training data
class MajorityClassModel:
    # initialize model with training labels
    def __init__(self, labels):
        # set majority class as the label that has the highest occurence in the list
        self.__majorityClass = max(labels,key=labels.count)

    # assigns majority class label to input data
    def test(self, input):
        return self.__majorityClass
    
    # evaluates model based on lists of test data and labels 
    def evaluate(self, data, labels):
        if len(data) != len(labels):
            raise Exception("data and label lists are not the same size")
        
        # calculate accuracy of model
        correct = 0
        for i in range(0, len(data)):
            if self.test(data[i]) == labels[i]:
                correct += 1
        
        # return accuracy
        return correct / len(data)
        