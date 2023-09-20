from sklearn.feature_extraction.text import CountVectorizer


class DialogueSystem:


    def __init__(self,lr_model,insts_train):
        self.classifier = lr_model
        self.vectorizer = CountVectorizer().fit(insts_train)
        
    def state_transition(self,utter,current_state):
        utter_vectorized = self.vectorizer.transform([utter])
        new_state = self.classifier.predict(utter_vectorized)
        print(new_state)
        return new_state
    
    def dialogue(self,suggestions,extract):
        current_state = ''
        
        print("Welcome to G11's restaurant system. You can ask for restaurants by area , price range or food type. How may I help you?")
        
        while current_state not in ['bye']:
            user_utterance = input()
            current_state = self.state_transition(user_utterance,current_state)
            