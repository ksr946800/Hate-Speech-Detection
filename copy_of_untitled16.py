
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

df = pd.read_csv(r"C:\Users\satya\Desktop\Hate speech detection\labeled_data.csv")

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(remove_punctuations)
df['tweet'] = df['tweet'].apply(remove_stopwords)

max_words = 10000
max_len = 50

# Tokenization
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(df['tweet'])

# Padding sequences
X = tokenizer.texts_to_sequences(df['tweet'])
X = pad_sequences(X, maxlen=max_len)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    Bidirectional(LSTM(16)),
    Dense(512, activation='relu', kernel_regularizer='l1'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, df['class'], test_size=0.2, random_state=42)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

app = Flask(__name__)

def preprocess_input(text):
    text = text.lower()
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    return text

def predict_hate_speech(text):
    text = preprocess_input(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(sequence)[0]
    predicted_class = np.argmax(prediction)
    return predicted_class

def home():
    return render_template('index.html')

def predict():
    user_input = request.form['text']
    prediction = predict_hate_speech(user_input)
    if prediction == 0:
        response = "Hate Speech"
    elif prediction == 1:
        response = "Offensive Language"
    else:
        response = "Neither"
    return render_template('result.html', input_text=user_input, prediction=response)

if __name__ == '__main__':
    app.run(debug=True)