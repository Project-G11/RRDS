import csv
from random import randrange

class SuggestRestaurants:

    def __init__(self):
        self.filename = 'restaurant_info.csv'

    def suggest(self, restaurants, number):
        if number == 0:
            print("There are no restaurants that conform to the requirements.")
            return
        if number == 1:
            i = 0
        else:
            i = randrange(number-1)
        print("A restaurant that conforms to the requirements is", restaurants[i][0], "with phone number", restaurants[i][1], ", address", restaurants[i][2], "and postal code", restaurants[i][3])
        restaurants.pop(i)
        return restaurants

    def findrestaurants(self, price, area, food, restaurants):
        with open(self.filename) as file:
            reader = csv.reader(file)
            number = 0
            for row in reader:
                try:
                    if (price == "any" or price == row[1]) and (area == "any" or area == row[2]) and (food == "any" or food == row[3]):
                        restaurants.append([row[0], row[4], row[5], row[6]])
                        number += 1
                except:
                    continue
        return restaurants, number