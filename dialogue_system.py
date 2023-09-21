from sklearn.feature_extraction.text import CountVectorizer
from enum import Enum


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
    
class Responses(Enum):
    system_uterances = {
            'noarea':'What part of town do you have in mind?',
            'nofoodtype': 'What kind of food would you like?',
            'nopricerange': 'Would you like something in the cheap , moderate , or expensive price range?',
            'confirmpricerange':' restaurant is in the pricerange',
            'confirmarea': ' restaurant is in the part of town',
            'confirmfoodtype': ' restaurant is serving food',
            'confirmfoodtype': 'You are looking for a restaurant right?',
            'confirmarea': 'You are looking for a restaurant in the part of town right?',
            'confirmpricerange': 'You are looking for a restaurant in the pricerange right?' ,
            'goodbye': 'Goodbye!',
            'phone': 'The phone number for is ',
            'address': ' is on ',
            'suggestion': ' is a nice place in the of town '
            }
    dialogue_acts = ['ack','affirm','bye','confirm','deny','hello','inform','negate','null','repeat','reqalts','reqmore','request','restart','thankyou']
    


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
        
    def classify_intent(self,utter):
        utter_vectorized = self.vectorizer.transform([utter])
        new_state = self.classifier.predict(utter_vectorized)
        return new_state
        
    def state_transition(self,current_state,user_utterance):
        # Predicting the user's intent
        user_intent = self.classify_intent(user_utterance)
        print(user_intent)
        
        # STATE TRANSITION
        if current_state == DialogState.INIT:
            if user_intent in ['null']:
                next_state = DialogState.ASK_FOOD_TYPE
                system_response = Responses.system_uterances['nofoodtype']
            elif user_intent in ['hello', 'inform', 'affirm']:
                
                next_state = DialogState.INIT
                system_response = "I'm sorry, I didn't understand that. Please start by greeting."

        
        
        
        return user_intent
    
    def run_dialogue(self,suggestions):
        # Default state
        current_state = DialogState.INIT
         
        # Iterative dialogue until user ends the conversation       
        while current_state != DialogState.END:
            user_utterance = input("> ").lower()
            current_state, system_response = self.state_transition(current_state, user_utterance)
            
            