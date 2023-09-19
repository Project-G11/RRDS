import csv
from random import randrange
import pandas as pd

class SuggestRestaurants:

    def __init__(self):
        self.restaurants = pd.read_csv('restaurant_info.csv')

    def findrestaurants(self, area, food, price):
        rest = self.restaurants
        
        if price == 'any' and area == 'any' and food == 'any':
            suggestions = rest.copy()
        elif price == 'any' and area == 'any':
            suggestions = rest[rest['food'] == food]
        elif price == 'any' and food == 'any':
            suggestions = rest[rest['area'] == area]
        elif area == 'any' and food == 'any':
            suggestions = rest[rest['pricerange'] == price]
        elif price == 'any':
            suggestions = rest[(rest['area'] == area) & (rest['food'] == food)]
        elif area == 'any':
            suggestions = rest[(rest['pricerange'] == price) & (rest['food'] == food)]
        elif food == 'any':
            suggestions = rest[(rest['pricerange'] == price) & (rest['area'] == area)]
        else:
            suggestions = rest[(rest['pricerange'] == price) & (rest['area'] == area) & (rest['food'] == food)]
        
        if not suggestions.empty:
            return suggestions
        else:
            return None
    
    
    def suggest(self, area, food, price): 
        return self.findrestaurants(area, food, price)