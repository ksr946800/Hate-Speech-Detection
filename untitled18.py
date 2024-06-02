
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from flask import Flask, render_template, request, jsonify
from numpy import ndarray
import re
# Text Pre-processing libraries
import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Tensorflow imports to build the model.
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

combined_df = pd.read_csv(r'C:\Users\satya\Desktop\Hate speech detection\labeled_data.csv')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\u0C80-\u0CFF\s]', '', text)
    # Tokenization (assuming whitespace tokenization for Kannada)
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming (optional, based on performance)
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back to text
    text = ' '.join(tokens)
    return text

combined_df['tweet'] = combined_df['tweet'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(combined_df['tweet'], combined_df['class'], test_size=0.2, random_state=42)

max_words = 10000
max_len = 50

tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_len)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)

input_layer = Input(shape=(max_len,))

# Embedding layer
embedding = Embedding(input_dim=max_words, output_dim=32, input_length=max_len)(input_layer)

# LSTM layer
lstm = Bidirectional(LSTM(16))(embedding)

# Dense layers for classification
dense = Dense(512, activation='relu', kernel_regularizer='l1')(lstm)
dense = BatchNormalization()(dense)
dense = Dropout(0.3)(dense)
output = Dense(1, activation='sigmoid')(dense)

# Create the model
model = Model(inputs=input_layer, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
early_stopping = EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

y_pred = model.predict(X_test)

def censor_hate_words(text_sequence, prediction):
    if prediction >= 0.5:  # Threshold for hate speech
        # Convert tokenized sequence back to text
        text = tokenizer.sequences_to_texts([text_sequence])[0]
        # Define list of hate words (replace with actual hate words)
        hate_words = ['hate_word1', 'hate_word2', 'hate_word3']
        for word in hate_words:
            text = text.replace(word, '*' * len(word))  # Replace hate words with asterisks
    else:
        text = tokenizer.sequences_to_texts([text_sequence])[0]  # Convert tokenized sequence back to text
    return text

# Example: Censoring hate words in the first 10 test samples
for i in range(10):
    print("Original Text:")
    print(X_test[i])
    print("Censored Text:")
    print(censor_hate_words(X_test[i], y_pred[i]))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()