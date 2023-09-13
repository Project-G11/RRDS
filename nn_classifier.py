import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam  
import matplotlib.pyplot as plt

class NNClassifier:
    
    def __init__(self, insts_train, insts_test, labels_train, labels_test):
        # Initialize the NNClassifier with training and testing data
        self.model = self.build_model(insts_train, insts_test, labels_train, labels_test)
        
    def build_model(self, insts_train, insts_test, labels_train, labels_test):
        # Define the maximum words in a sentence (the model needs to have that configured)
        max_words = 150
        # All the unique names of the labels(classes). They are 15
        classes = np.unique(labels_train)
        # Convert all the arrays in numpy arrays to manipulate the data easier
        insts_train = np.array(insts_train)
        insts_test = np.array(insts_test)
        labels_train = np.array(labels_train)
        labels_test = np.array(labels_test)
        # Tokenize each sentence and add padding based on the max_words per sentence.
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(insts_train)
        
        sequences_train = tokenizer.texts_to_sequences(insts_train)
        X_train = pad_sequences(sequences_train, maxlen=max_words)
        
        sequences_test = tokenizer.texts_to_sequences(insts_test)
        X_test = pad_sequences(sequences_test, maxlen=max_words)
        
        #-----------------------------------------------------------------
        def map_labels(labels_train):
            unique_labels = np.unique(labels_train)
            label_mapping={}
            i=0
            for label in unique_labels:
                label_mapping[label] = i
                i+=1
            labels_train_int = [label_mapping[label] for label in labels_train]
            return labels_train_int

        labels_train_int = map_labels(labels_train)
        y_train = keras.utils.to_categorical(labels_train_int, num_classes=len(classes))
        
        labels_test_int = map_labels(labels_test)
        y_test = keras.utils.to_categorical(labels_test_int, num_classes=len(classes))

        #-----------------------------------------------------------------
        model = keras.Sequential()
        print(insts_train.shape)
        print(len(classes))
        model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_words))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        # model.add(keras.layers.Dropout(0.25))
        model.add(Dense(len(classes), activation='softmax'))  

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.train(model, X_train, y_train, X_test, y_test)
    
    
    def train(self, model, X_train, y_train,  X_test, y_test):
        history = model.fit(X_train, y_train, epochs=6, batch_size=8, validation_data=(X_test, y_test))
        self.evaluation(X_train, y_train, model, history)
        
    def evaluation(self, X_train, y_train, model, history):
        loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
        print("Feedforward Neural Netrowk classifier accuracy:",history.history['val_accuracy'][-1])
        self.show_results(history)
        
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