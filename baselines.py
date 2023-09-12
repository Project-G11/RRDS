# model that assigns labels to data based on the majority class of the training data
class MajorityClassModel:
    # initialize model with training labels
    def __init__(self, labels):
        # set majority class as the label that has the highest occurence in the list
        self.majorityClass = max(labels,key=labels.count)

    # assigns majority class label to input data
    def test(self, input):
        return self.majorityClass
    
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
        
# model that assigns labels to data based on keyword matching rules
class KeywordMatchingModel(MajorityClassModel):
    def __init__(self, labels):
        super().__init__(labels)

    def test(self, input):
        if "bye" in input:
            return "bye"
        elif "thank" in input:
            return "thankyou"
        elif "hi " in input or "hello" in input or "helo " in input:
            return "hello"
        else: 
            return self.majorityClass
    
    