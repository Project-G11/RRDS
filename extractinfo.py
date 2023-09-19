import Levenshtein
import pandas as pd

class ExtractInformation:

    # initialize a dictionary with all keywords
    def __init__(self):
        self.keywords = {}
        self.info = {}
        df = pd.read_csv('data/restaurant_info.csv')
        pricerange = df.pricerange.unique().tolist()
        area = df.area.unique().tolist()
        food = df.food.unique().tolist()
        self.keywords["pricerange"] = pricerange
        self.keywords["area"] = area
        self.keywords["food"] = food

    def findwords(self, input):
        words = input.lower().split()
        print(words)
        for word in words:
            if len(word)>3:
                for i in self.keywords:
                    if self.info.get(i) == None:
                        for j in self.keywords[i]:
                            if str(word) == str(j):
                                self.info[i] = j
                            elif Levenshtein.distance(str(word), str(j)) < 2:
                                self.info[i] = j
        return self.info