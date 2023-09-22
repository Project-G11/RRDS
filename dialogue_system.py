from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum
import numpy as np
from transformers import BertTokenizer
import pandas as pd
import os



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
    
    def __init__(self,lr_model,insts_train,extract):
        # Initialize classifier, vectorizer and extract
        self.classifier = lr_model
        self.vectorizer = CountVectorizer().fit(insts_train)
        self.extract = extract
        
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
            'notunderstand': "I didn't understand your request. Please repeat.", 
            'phone': 'The phone number for is ',
            'address': ' is on ',
            'suggestion': 'is a nice place in the side of town'
            }
        self.dialogue_acts = ['ack','affirm','bye','confirm','deny','hello','inform','negate','null','repeat','reqalts','reqmore','request','restart','thankyou']
        
        # Initialize food, place and price
        
        # Open the restaurants file and extract details for later use
        self.restaurants = pd.read_csv('data/restaurant_info.csv')
        
        # Initialize info
        self.info = {}
        
        
        
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
        
        
        # STATE TRANSITION
        #-- First Reply --
        if current_state == DialogState.INIT:
            if user_intent in ['null']:
                next_state = DialogState.ASK_FOOD_TYPE
                system_response = self.system_uterances['nofoodtype']
            elif user_intent in ['hello', 'inform', 'affirm','confirm','request']:
                self.info = self.extract.findwords(user_utterance)
                print(self.info)
                if not 'food' in self.info:
                    print("~No food~")
                    next_state = DialogState.ASK_FOOD_TYPE
                    system_response = self.system_uterances['nofoodtype']
                else:
                    if not 'area' in self.info:
                        print("~No area~")
                        next_state = DialogState.ASK_AREA
                        system_response = self.system_uterances['noarea']
                    else:
                        if not 'pricerange' in self.info:
                            print("~No pricerange~")
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_uterances['nopricerange']
                        else:
                            next_state = DialogState.SUGGEST
                            suggestion = self.suggestions.suggest(self.info['area'],self.info['food'],self.info['pricerange']).iloc[0]
                            system_response = self.system_uterances['suggestion']
                            if self.info['pricerange'] != None:
                                system_response = '{} {} {} {} and it is in the {} price range.'.format(suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:], suggestion.pricerange)
                            else:
                                system_response = '{} {} {} {}'.format(suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:])
                                
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
            if 'food' not in self.info:
                # Check if user mentioned a food preference
                food_info = self.extract.findwords(user_utterance)
                if food_info is not None and 'food' in food_info:
                    self.info['food'] = food_info['food']
                else:
                    if current_state == DialogState.ASK_FOOD_TYPE:
                        self.info['food'] = None
                    else:
                        # Ask if the user wants any food type
                        next_state = DialogState.ASK_FOOD_TYPE
                        system_response = self.system_uterances['anyfood']
            
            if 'food' in self.info and 'area' not in self.info:
                # Check if user mentioned an area preference
                area_info = self.extract.findwords(user_utterance)
                if area_info is not None and 'area' in area_info:
                    self.info['area'] = area_info['area']
                else:
                    if current_state == DialogState.ASK_AREA:
                        self.info['area'] = None
                    else:
                        # Ask if the user wants any area
                        next_state = DialogState.ASK_AREA
                        system_response = self.system_uterances['anyplace']
            
            if 'food' in self.info and 'area' in self.info and 'pricerange' not in self.info:
                # Check if user mentioned a price range preference
                pricerange_info = self.extract.findwords(user_utterance)
                if pricerange_info is not None and 'pricerange' in pricerange_info:
                    self.info['pricerange'] = pricerange_info['pricerange']
                else:
                    if current_state == DialogState.ASK_PRICE_RANGE:
                        self.info['pricerange'] = None
                    else:
                        # Ask if the user wants any price range
                        next_state = DialogState.ASK_PRICE_RANGE
                        system_response = self.system_uterances['anyprice']
            
            if 'food' in self.info and 'area' in self.info and 'pricerange' in self.info:
                # All slots filled, suggest a restaurant
                next_state = DialogState.SUGGEST
                suggestion = self.suggest()
                system_response = self.system_uterances['suggestion']
                if self.info['pricerange'] != None:
                    system_response = '{} {} {} {} and it is in the {} price range.'.format(
                        suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:], suggestion.pricerange)
                else:
                    system_response = '{} {} {} {}'.format(suggestion.restaurantname, system_response[:18], suggestion.area, system_response[23:])

                
        print("Next state: ", next_state, " System response: ", system_response)
        return next_state, system_response, self.info
    
    def suggest(self):
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
            current_state, system_response, info = self.state_transition(current_state, user_utterance)
   
            self.print_response(system_response)
            
            if current_state == DialogState.END:
                break
            
            
            
            