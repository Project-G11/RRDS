from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum
import numpy as np
from transformers import BertTokenizer
import Levenshtein
import pandas as pd
import os
import random
import csv
import time



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
        self.init_slots()
        # Initialize system responses
        self.init_system_responses()
        # List of dialogue acts
        self.dialogue_acts = ['ack','affirm','bye','confirm','deny','hello','inform','negate','null','repeat','reqalts','reqmore','request','restart','thankyou']        
        # Open the restaurants file and extract details for later use
        self.restaurants = pd.read_csv('data/restaurant_info_new.csv')
        # Add properties to restaurant info with random values, only used once to create restaurant_info_new.csv
        '''
        self.restaurants["quality"] = np.random.choice(["good food", "bad food"], self.restaurants.shape[0])
        self.restaurants["crowdedness"] = np.random.choice(["busy", "not busy"], self.restaurants.shape[0])
        self.restaurants["staylength"] = np.random.choice(["long stay", "short stay"], self.restaurants.shape[0])
        self.restaurants.to_csv('data/restaurant_info_new.csv', quoting = csv.QUOTE_ALL, index = False)
        '''
        # Get keywords to be able to extract user input information
        self.init_keywords()
        # Initialize info
        self.info = {}
        # How many times will the system persist
        self.tries = {'food':2, 'area':2, 'pricerange':2}
        # Configurability using Levenshtein distance
        self.levenshtein_dist = True
        # Configurability using all caps
        self.all_caps = False
        # Configurability checking Levenshtein match correctness
        self.levenshtein_match = False
        # Configurability delay in system response
        self.delay = False
        
    def init_system_responses(self):
        self.system_responses = {
            'greet': "Hello, welcome to the G11's restaurant system? You can ask for restaurants by area, price range, or food type. How may I help you?",
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
            'suggestion': 'is a(n)restaurant in theside of town',
            'noplace': 'Unfortunately, there is no such place.',
            'additionalreqs': 'Do you have additional requirements?'
        }
        
    def init_slots(self):
        self.food_type = None
        self.area = None
        self.price_range = None
        
    def init_keywords(self):
        self.keywords = {
            "pricerange": [str(kw).lower().strip() for kw in self.restaurants.pricerange.unique().tolist()],
            "area": [str(kw).lower().strip() for kw in self.restaurants.area.unique().tolist()],
            "food": [str(kw).lower().strip() for kw in self.restaurants.food.unique().tolist()],
        }
        
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
        
        def generate_suggestion_response():
            next_state = DialogState.SUGGEST
            suggestion, addresp = self.suggest()
            try:
                system_response = self.system_responses['suggestion']
                if self.info['pricerange'] is not None:
                    system_response = '{} {} {} {} {} {} and it is in the {} price range.'.format(
                        suggestion.restaurantname, system_response[:7], suggestion.food, system_response[7:24], suggestion.area, system_response[24:], suggestion.pricerange) + addresp
                    next_state = DialogState.END
                else:
                    system_response = '{} {} {} {}'.format(suggestion.restaurantname, system_response[:7], suggestion.food, system_response[7:24], suggestion.area, system_response[24:]) + addresp
                    next_state = DialogState.END
            except:
                system_response = self.system_responses['noplace']
                next_state = DialogState.END
                
            return system_response, next_state

        
        # STATE TRANSITION
        #-- First Reply --
        if current_state == DialogState.INIT:
            if user_intent in ['null']:
                next_state = DialogState.ASK_FOOD_TYPE
                system_response = self.system_responses['nofoodtype']
            elif user_intent in ['hello', 'inform', 'affirm','confirm','request']:
                self.info, found = self.findwords(self.info, user_utterance, self.keywords)
                    
                if not 'food' in self.info:
                    next_state = DialogState.ASK_FOOD_TYPE
                    system_response = self.system_responses['nofoodtype']
                    self.tries['food'] = self.tries['food'] -1
                else:
                    if not 'area' in self.info:
                        next_state = DialogState.ASK_AREA
                        system_response = self.system_responses['noarea']
                        self.tries['area'] = self.tries['area'] -1
                    else:
                        if not 'pricerange' in self.info:
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_responses['nopricerange']
                            self.tries['pricerange'] = self.tries['pricerange'] -1
                        else:
                           system_response, next_state = generate_suggestion_response()
                                
            elif user_intent in ['bye']:
                next_state = DialogState.END
                system_response = self.system_responses['goodbye']
            elif user_intent in ['thankyou']:
                next_state = DialogState.THANK_YOU
                system_response = self.system_responses['thankyou']
            else:
                next_state = DialogState.NOT_UNDERSTAND
                system_response = self.system_responses['notunderstand']
        
        #-- Other Replies --
        else:
            new_info, found = self.findwords(new_info, user_utterance, self.keywords)
            if not found:
                system_response = self.system_responses['notunderstand']
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
                            system_response = self.system_responses['anyfood']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['food'] = self.tries['food'] -1
                        else:
                            next_state = DialogState.ASK_FOOD_TYPE
                            system_response = self.system_responses['nofoodtype']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['food'] = self.tries['food'] -1
                else:
                    if 'food' in new_info:
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
                            system_response = self.system_responses['anyplace']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['area'] = self.tries['area'] -1
                        else:
                            next_state = DialogState.ASK_AREA
                            system_response = self.system_responses['noarea']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['area'] = self.tries['area'] -1
                elif 'food' in self.info and 'area' in self.info:
                    if 'area' in new_info:
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
                            system_response = self.system_responses['anyprice']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['pricerange'] = self.tries['pricerange'] -1
                        else:
                            next_state = DialogState.ASK_PRICE_RANGE
                            system_response = self.system_responses['nopricerange']
                            if not (set(new_info.keys()) & set(self.info.keys())):
                                self.tries['pricerange'] = self.tries['pricerange'] -1
                
                if 'food' in self.info and 'area' in self.info and 'pricerange' in self.info:
                    system_response, next_state = generate_suggestion_response()
                    
        return next_state, system_response
    
    # Finds keywords that provide information for the system in the user input
    def findwords(self, info, words, keywords):
        words = words.lower().split()
        found = False
        for word in words:
            if len(word)>3:
                for i in keywords:
                    for j in keywords[i]:
                        if self.levenshtein_dist:
                            if self.levenshtein_match:
                                if word == j:
                                    info[i] = j
                                    found = True
                                if Levenshtein.distance(word, j) < 2:
                                    self.print_response("Did you mean " + j + " by " + word + "?")
                                    userinput = input(">>> ").lower()
                                    user_intent = self.classify_intent(userinput)
                                    if user_intent in ['affirm']:
                                        info[i] = j
                                        found = True
                            else:
                                if word == j or Levenshtein.distance(word, j) < 2:
                                    info[i] = j
                                    found = True
                        else:
                            if word == j:
                                info[i] = j
                                found = True
        return info, found
    
    # Provides recommended restaurants based on the preferences of the user
    def suggest(self):
        # ask for additional requirements
        self.print_response(self.system_responses['additionalreqs'])
        userinput = input(">>> ").lower()
        intent = self.classify_intent(userinput)
        suggestions = self.getSuggestion()
        # if user doesn't give additional requirements, get the first restaurant from the list
        if intent == 'negate':
            return suggestions.iloc[0], ""
        else:
            # extract user input requirement
            req_options = {"requirements": ["romantic", "touristic", "children", "assigned seats"]}
            info, found = self.findwords(self.info, userinput, req_options)
            # if requirement is extracted, filter existing suggestions based on requirement
            if found:
                newsug, addreq = self.reasonOnReqs(info["requirements"], suggestions)
                if not newsug.empty:
                    return newsug.iloc[0], addreq
            
            # User asked for address
            request_address = {"address": ["address"]}
            request_phone = {"phone": ["phone"]}
            info, found_address = self.findwords(self.info, userinput, request_address)
            info, found_phone = self.findwords(self.info, userinput, request_phone)
            if found_address or found_phone:
                extra_info = self.get_address_phone(info,suggestions)
                return suggestions.iloc[0], extra_info

            

            # if doesn't understand requirement or no restaurant exist with given requirement
            return suggestions.iloc[0], " Unfortunately, I could not find any restaurants with your additional requirement."

    # Provides the user with the address and phone number of the recommended restaurant if these are availabale
    def get_address_phone(self,information,suggestions):
        print(information)
        if 'address' in information and not 'phone' in information:
            return " The address is: " + suggestions.iloc[0]['addr'] + "."
        elif 'phone' in information and 'address' not in information:
            return " The phone is: " + suggestions.iloc[0]['phone'] + "."
        else:
            return " Address: " + suggestions.iloc[0]['addr']+ "." + " Phone: " + suggestions.iloc[0]['phone'] + "."
            
    # Finds restaurants that correspond to the wishes of the user        
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
            return rest  
        
    def reasonOnReqs(self, req, sugs):
        if req == "touristic":
            return sugs[(sugs["pricerange"] == "cheap") & (sugs["quality"] == "good food") & sugs["food"] != "romanian"] \
            , " This restaurant is touristic because the food is cheap and good"
        elif req =="romantic":
            return sugs[(sugs["staylength"] == "long stay") & (sugs["crowdedness"] == "not busy")] \
            , " The restaurant is romantic because it allows you to stay for a long time."
        elif req =="children":
            return sugs[sugs["staylength"] == "short stay"] \
            , " The restaurant is good for bringing children because people usually stay a short time."
        elif req =="assigned seats":
            return sugs[sugs["crowdedness"] == "busy"] \
            , " The waiter decides where you sit because the restaurant is busy."

    # Prints system output   
    def print_response(self,response):
        if self.delay:
            time.sleep(3)
        if self.all_caps:
            print('> System:',response.upper())
        else:
            print('> System:',response)
    
    def run_dialogue(self):
        # Clear console
        os.system('cls')
        # Default state
        current_state = DialogState.INIT
        self.print_response(self.system_responses['greet'])
        # Iterative dialogue until user ends the conversation       
        while current_state not in [DialogState.END, DialogState.GOODBYE]:
            user_utterance = input(">>> ").lower()
            current_state, system_response = self.state_transition(current_state, user_utterance)
   
            self.print_response(system_response)
            
            if current_state == DialogState.END:
                break
            