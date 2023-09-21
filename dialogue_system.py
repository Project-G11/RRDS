from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum
import numpy as np
from transformers import BertTokenizer



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
    GOODBYE = 10
    SUGGEST = 11
    END = 12
    

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
        self.missing_slots = []
        self.missing_slots.append('food_type')
        self.missing_slots.append('area')
        self.missing_slots.append('price_range')
        print(self.missing_slots)
        
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
            'confirmpricerange': 'You are looking for a restaurant in the pricerange right?' ,
            'goodbye': 'Goodbye!',
            'phone': 'The phone number for is ',
            'address': ' is on ',
            'suggestion': ' is a nice place in the of town '
            }
        self.dialogue_acts = ['ack','affirm','bye','confirm','deny','hello','inform','negate','null','repeat','reqalts','reqmore','request','restart','thankyou']
        
    def preprocess_sentence(self,sentence, max_words):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        token_ids = tokenizer.encode(sentence, max_length=max_words, truncation=True, padding='max_length')
        return np.array(token_ids)

    def classify_intent(self,utter):
        utter_vectorized = self.preprocess_sentence(utter,150)
        new_state = self.classifier.predict(np.array([utter_vectorized]))
        print(new_state.shape)
        return new_state
        
    def state_transition(self,current_state,user_utterance):
        # Predicting the user's intent
        user_intent = self.classify_intent(user_utterance)
        print(user_intent)
        # Initialize default system_response
        system_response = 'null'
        
        # STATE TRANSITION
        if current_state == DialogState.INIT:
            if user_intent in ['null']:
                next_state = DialogState.ASK_FOOD_TYPE
                system_response = self.system_uterances['nofoodtype']
            elif user_intent in ['hello', 'inform', 'affirm']:
                
                next_state = DialogState.INIT
                system_response = "I'm sorry, I didn't understand that. Please start by greeting."

        
        return current_state, system_response
    
    
    def run_dialogue(self,suggestions):
        # Default state
        current_state = DialogState.INIT
        print(self.system_uterances['greet'])
        # Iterative dialogue until user ends the conversation       
        while current_state not in [DialogState.END, DialogState.GOODBYE]:
            user_utterance = input("> ").lower()
            current_state, system_response = self.state_transition(current_state, user_utterance)
            
            