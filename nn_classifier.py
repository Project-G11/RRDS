import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer
import pickle


class NNClassifier:
    
    def __init__(self, insts_train, insts_test, labels_train, labels_test, no_duplicates):
        # Initialize the NNClassifier with training and testing data
        self.model = self.build_model(insts_train, insts_test, labels_train, labels_test, no_duplicates)
        
    def build_model(self, insts_train, insts_test, labels_train, labels_test, no_duplicates):
        # Define the maximum words in a sentence (the model needs to have that configured)
        max_words = 150
        # All the unique names of the labels(classes). They are 15
        classes = np.unique(labels_train)
        # Convert all the arrays in numpy arrays to manipulate the data easier
        
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        
        # Function that uses pre-trained tokenizer to tokenize sentences and labels
        def tokenize_sentences(list,max_words):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            token_ids = [tokenizer.encode(sentence, max_length=max_words, truncation=True, padding='max_length') for sentence in list]
            return np.array(token_ids)
            
        X_train = tokenize_sentences(insts_train,max_words)
        X_test = tokenize_sentences(insts_test,max_words)

        def map_labels(labels,length):
            unique_labels = np.unique(labels)
            label_mapping={}
            i=0
            for label in unique_labels:
                label_mapping[label] = i
                i+=1
            labels_int = [label_mapping[label] for label in labels]
            one_hot_labels = keras.utils.to_categorical(labels_int, num_classes=length)
            return one_hot_labels

        y_train = map_labels(labels_train,len(classes))        
        y_test = map_labels(labels_test,len(classes))
        
        print(X_train.shape)
        
        model = keras.Sequential()
        model.add(Embedding(input_dim=X_train.max()+1, output_dim=50, input_length=max_words))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(classes), activation='softmax'))  

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.train(model, X_train, y_train, X_test, y_test, no_duplicates)
    
    
    def train(self, model, X_train, y_train,  X_test, y_test, no_duplicates):
        batch_size = 32 if no_duplicates else 128
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=12, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])
        
        #Creating a file for our model
        with open('models/ffnn_model', 'wb') as f:
            pickle.dump(model,f)
        
        self.evaluation(X_train, y_train, model, history)
        
    def evaluation(self, X_train, y_train, model, history):
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
        print("Feedforward Neural Netrowk classifier accuracy:",history.history['val_accuracy'][-1])
        # self.show_results(history)
        
    def show_results(self, history):
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']

        # Plot training and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(training_accuracy, label='Training Accuracy')
        plt.plot(validation_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()