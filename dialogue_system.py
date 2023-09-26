from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum
import numpy as np
from transformers import BertTokenizer
import Levenshtein
import pandas as pd
import os
import random



class DialogState(Enum):
    INIT = 0
    ASK_FOOD_TYPE = 1
    ASK_AREA = 2
    ASK_PRICE_RANGE = 3
    CONFIRM_FOOD_TYPE = 4
    CONFIRM_AREA = 5
    CONFIRM_PRICE_RANGE = 6
    PHONE = 7
    ADDRESS = 8
    THANK_YOU = 9
    NOT_UNDERSTAND = 10
    GOODBYE = 11
    SUGGEST = 12
    END = 13
    

class DialogueSystem:
    
    def __init__(self,lr_model,insts_train):
        # Initialize classifier, vectorizer and extract
        self.classifier = lr_model
        self.vectorizer = CountVectorizer().fit(insts_train)
        
        # Initialize slots to None
        self.food_type = None
        self.area = None
        self.price_range = None
        
        # Initialize system responses
        self.system_uterances = {
            'greet': "Hello , welcome to the G11's restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?",
            'noarea':'What part of town do you have in mind?',
            'nofoodtype': 'What kind of food would you like?',
            'nopricerange': 'Would you like something in the cheap , moderate , or expensive price range?',
            'affirmpricerange':' restaurant is in the pricerange',
            'affirmarea': ' restaurant is in the part of town',
            'affirmfoodtype': ' restaurant is serving food',
            'confirmfoodtype': 'You are looking for a restaurant right?',
            'confirmarea': 'You are looking for a restaurant in the part of town right?',
            'confirmpricerange': 'You are looking for a restaurant in the pricerange right?',
            'anyfood': 'You are looking for a restaurant serving any kind of food?',
            'anyplace': 'You are looking for a restaurant at any place?',
            'anyprice': 'You are looking for a restaurant with any price?',
            'goodbye': 'Goodbye!',
            'thankyou': "You're welcome",
            'notunderstand': "Sorry, I didn't understand your request. Please repeat.", 
            'phone': 'The phone number for is ',
            'address': ' is on ',
            'suggestion': 'is a nice place in the side of town',
            'noplace': 'Unfortunately, there is no such place.',
            'additionalreqs': 'Do you have additional requirements?'
            }
        self.dialogue_acts = ['ack','affirm','bye','confirm','deny','hello','inform','negate','null','repeat','reqalts','reqmore','request','restart','thankyou']
        
        # Initialize food, place and price
        
        # Open the restaurants file and extract details for later use
        self.restaurants = pd.read_csv('data/restaurant_info.csv')

        # Add properties to restaurant info with random values
        self.restaurants["quality"] = np.random.choice(["good food", "bad food"], self.restaurants.shape[0])
        self.restaurants["crowdedness"] = np.random.choice(["busy", "not busy"], self.restaurants.shape[0])
        self.restaurants["staylength"] = np.random.choice(["long stay", "short stay"], self.restaurants.shape[0])

        # Get keywords to be able to extract user input information
        self.keywords = {}
        pricerange = self.restaurants.pricerange.unique().tolist()
        area = self.restaurants.area.unique().tolist()
        food = self.restaurants.food.unique().tolist()
        self.keywords["pricerange"] = [str(kw).lower().strip() for kw in pricerange]
        self.keywords["area"] = [str(kw).lower().strip() for kw in area]
        self.keywords["food"] = [str(kw).lower().strip() for kw in food]
        
        # Initialize info
        self.info = {}
        
        # Initialize food, area and price tries
        self.tries = {'food':2, 'area':2, 'pricerange':2}
        
    def preprocess_sentence(self,sentence, max_words):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        token_ids = tokenizer.encode(sentence, max_length=max_words, truncation=True, padding='max_length')
        return np.array(token_ids)

    def classify_intent(self,utter):
        try:
            utter_vectorized = self.preprocess_sentence(utter,150)
            new_state = self.dialogue_acts[np.argmax(self.classifier.predict(np.array([utter_vectorized])))]
        except:
            utter_vectorized = self.vectorizer.transform([utter])
            new_state = self.classifier.predict(utter_vectorized)
        return new_state
        
    def state_transition(self,current_state,user_utterance):
        # Predicting the user's intent
        user_intent = self.classify_intent(user_utterance)
        print(user_intent)
        # Initialize default system_response and next_state
        system_response = 'null'
        next_state = ''
        new_info = {}
        
        # STATE TRANSITION
        #-- First Reply --
        if current_state == DialogState.INIT:
            if user_intent in ['null']:
                next_state = DialogState.ASK_FOOD_TYPE
                system_response = self.system_uterances['nofoodtype']
            elif user_intent in ['hello', 'inform', 'affirm','confirm','request']:
                self.info, found = self.findwords(user_utterance)
                if not found:
                    system_response = self.system_uterances['notunderstand']
                    return next_state, system_response
                    
                print(self.info)
                if not 'food' in self.info:
                    print("~No food~")
                    next_state = DialogState.ASK_FOOD_TYPE
                    system_response = self.system_uterances['nofoodtype']
                    self.tries['food'] = self.tries['food'] -1
                else:
                    if not 'area' in self.info:
                        print("~No area~")
                        next_state = DialogState.ASK_AREA
                        system_response = self.system_uterances['noarea']
                        self.tries['area'] = self.tries['area'] -1
                    else:
                        if not 'pricerange' in self.info:
                            print("~No pricerange~")
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_uterances['nopricerange']
                            self.tries['pricerange'] = self.tries['pricerange'] -1
                        else:
                            next_state = DialogState.SUGGEST
                            suggestion = self.suggest()
                            try:
                                system_response = self.system_uterances['suggestion']
                                if self.info['pricerange'] != None:
                                    system_response = '{} {} {} {} and it is in the {} price range.'.format(
                                        suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:], suggestion.pricerange)
                                    next_state = DialogState.END
                                else:
                                    system_response = '{} {} {} {}'.format(suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:])
                                    next_state = DialogState.END
                            except:
                                system_response = self.system_uterances['noplace']
                                next_state = DialogState.END
                                
            elif user_intent in ['bye']:
                next_state = DialogState.END
                system_response = self.system_uterances['goodbye']
            elif user_intent in ['thankyou']:
                next_state = DialogState.THANK_YOU
                system_response = self.system_uterances['thankyou']
            else:
                next_state = DialogState.NOT_UNDERSTAND
                system_response = self.system_uterances['notunderstand']
        
        #-- Other Replies --
        else:
            new_info, found = self.findwords(user_utterance)
            if not found:
                print("~~NOT FOUND~~")
                system_response = self.system_uterances['notunderstand']
                next_state = DialogState.NOT_UNDERSTAND
            else:
                if 'food' not in self.info:
                    if new_info is not None and 'food' in new_info:
                        self.info['food'] = new_info['food']
                    else:
                        if current_state == DialogState.ASK_FOOD_TYPE and self.tries['food'] == 0:
                            self.info['food'] = None
                        elif current_state == DialogState.ASK_FOOD_TYPE and self.tries['food'] == 1:
                            # Ask if the user wants any food type
                            next_state = DialogState.ASK_FOOD_TYPE
                            system_response = self.system_uterances['anyfood']
                            self.tries['food'] = self.tries['food'] -1
                        else:
                            next_state = DialogState.ASK_FOOD_TYPE
                            system_response = self.system_uterances['nofoodtype']
                            self.tries['food'] = self.tries['food'] -1
                else:
                    if 'food' in [new_info,self.info]:
                        print("FOOD CHANGES")
                        self.info['food'] = new_info['food']
                
                if 'food' in self.info and 'area' not in self.info:
                    if new_info is not None and 'area' in new_info:
                        self.info['area'] = new_info['area']
                    else:
                        if current_state == DialogState.ASK_AREA and self.tries['area'] == 0:
                            self.info['area'] = None
                        elif current_state == DialogState.ASK_AREA and self.tries['area'] == 1:
                            # Ask if the user wants any area
                            next_state = DialogState.ASK_AREA
                            system_response = self.system_uterances['anyplace']
                            self.tries['area'] = self.tries['area'] -1
                        else:
                            next_state = DialogState.ASK_AREA
                            system_response = self.system_uterances['noarea']
                            self.tries['area'] = self.tries['area'] -1
                elif 'food' in self.info and 'area' in self.info:
                    if 'area' in [new_info,self.info]:
                        print("AREA CHANGES")
                        self.info['area'] = new_info['area']
                
                if 'food' in self.info and 'area' in self.info and 'pricerange' not in self.info:
                    # Check if user mentioned a price range preference
                    if new_info is not None and 'pricerange' in new_info:
                        self.info['pricerange'] = new_info['pricerange']
                    else:
                        if current_state == DialogState.ASK_PRICE_RANGE and self.tries['pricerange'] == 0:
                            self.info['pricerange'] = None
                        elif current_state == DialogState.ASK_PRICE_RANGE and self.tries['pricerange'] == 1:
                            # Ask if the user wants any price range
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_uterances['anyprice']
                            self.tries['pricerange'] = self.tries['pricerange'] -1
                        else:
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_uterances['nopricerange']
                            self.tries['pricerange'] = self.tries['pricerange'] -1
                
                if 'food' in self.info and 'area' in self.info and 'pricerange' in self.info:
                    # All slots filled, suggest a restaurant
                    next_state = DialogState.SUGGEST
                    suggestion = self.suggest()
                    
                    try:
                        system_response = self.system_uterances['suggestion']
                        if self.info['pricerange'] != None:
                            system_response = '{} {} {} {} and it is in the {} price range.'.format(
                                suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:], suggestion.pricerange)
                            next_state = DialogState.END
                        else:
                            system_response = '{} {} {} {}'.format(suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:])
                            next_state = DialogState.END
                    except:
                        system_response = self.system_uterances['noplace']
                        next_state = DialogState.END
                    
        print("Next state: ", next_state, " System response: ", system_response)
        return next_state, system_response
    
    def findwords(self, input):
        words = input.lower().split()
        found = False
        # print(words)
        for word in words:
            if len(word)>3:
                for i in self.keywords:
                    for j in self.keywords[i]:
                        if word == j or Levenshtein.distance(word, j) < 2:
                            self.info[i] = j
                            found = True
        return self.info, found
    
    def suggest(self):
        # TODO: would be nicer if this was included in a state, but will let christos finish so I can more easily integrate it :)
        self.print_response(self.system_uterances['additionalreqs'])
        userinput = input(">>> ").lower()
        intent = self.classify_intent(userinput)
        suggestion = self.getSuggestion()
        if intent == 'negate':
            return suggestion
        else:
            # TODO: need to add the reasoning on additional requirement here
            return suggestion

    def getSuggestion(self):
        rest = self.restaurants
        area = self.info['area']
        food = self.info['food']
        price = self.info['pricerange']

        if (area):
            rest = rest[rest['area'] == area]
        if (food):
            rest = rest[rest['food'] == food]
        if (price):
            rest = rest[rest['pricerange'] == price]
            
        if not rest.empty:
            return rest.iloc[0]  
        
    def print_response(self,response):
        print('> System:',response)
    
    def run_dialogue(self):
        # Clear console
        os.system('cls')
        # Default state
        current_state = DialogState.INIT
        self.print_response(self.system_uterances['greet'])
        # Iterative dialogue until user ends the conversation       
        while current_state not in [DialogState.END, DialogState.GOODBYE]:
            user_utterance = input(">>> ").lower()
            current_state, system_response = self.state_transition(current_state, user_utterance)
   
            self.print_response(system_response)
            
            if current_state == DialogState.END:
                break
            