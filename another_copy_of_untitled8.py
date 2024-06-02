
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from flask import Flask, render_template, request, jsonify
import pickle
import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
import datetime
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
from keras import preprocessing
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, BatchNormalization, Dropout


nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\satya\Desktop\Hate speech detection\labeled_data.csv")
df.head(20)

df.shape

df.info()

plt.pie(df['class'].value_counts().values,
		labels = df['class'].value_counts().index,
		autopct='%1.1f%%')
plt.show()

# Lower case all the words of the tweet before any preprocessing
df['tweet'] = df['tweet'].str.lower()

# Removing punctuations present in the text
punctuations_list = string.punctuation
def remove_punctuations(text):
	temp = str.maketrans('', '', punctuations_list)
	return text.translate(temp)

df['tweet']= df['tweet'].apply(lambda x: remove_punctuations(x))
df.head()

def remove_stopwords(text):
	stop_words = stopwords.words('english')

	imp_words = []

	# Storing the important words
	for word in str(text).split():

		if word not in stop_words:

			# Let's Lemmatize the word as well
			# before appending to the imp_words list.

			lemmatizer = WordNetLemmatizer()
			lemmatizer.lemmatize(word)

			imp_words.append(word)

	output = " ".join(imp_words)

	return output


df['tweet'] = df['tweet'].apply(lambda text: remove_stopwords(text))
df.head()

def plot_word_cloud(data, typ):
    # Joining all the tweets to get the corpus
    email_corpus = " ".join(data['tweet'])

    plt.figure(figsize=(10, 10))

    # Forming the word cloud
    wc = WordCloud(max_words=100,
                   width=800,  # Increase the width for better visibility
                   height=400,  # Increase the height for better visibility
                   collocations=False).generate(email_corpus)

    # Plotting the wordcloud obtained above
    plt.title(f'WordCloud for {typ} emails.', fontsize=15)
    plt.axis('off')
    plt.imshow(wc)
    plt.show()

# Example usage with corrected code
plot_word_cloud(df[df['class'] == 2], typ='Neither')

class_2 = df[df['class'] == 2]
class_1 = df[df['class'] == 1].sample(n=3500)
class_0 = df[df['class'] == 0]

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

plt.pie(balanced_df['class'].value_counts().values,
		labels=balanced_df['class'].value_counts().index,
		autopct='%1.1f%%')
plt.show()

features = balanced_df['tweet']
target = balanced_df['class']

X_train, X_val, Y_train, Y_val = train_test_split(features,
												target,
												test_size=0.2,
												random_state=22)
X_train.shape, X_val.shape

Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)
Y_train.shape, Y_val.shape

max_words = 10000
max_len = 50

token = Tokenizer(num_words=max_words,
				lower=True,
				split=' ')

token.fit_on_texts(X_train)

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)


# Initialize the Tokenizer
token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(X_train)

# Generating token embeddings for training data
Training_seq = token.texts_to_sequences(X_train)
Training_pad = pad_sequences(Training_seq, maxlen=50, padding='post', truncating='post')

# Generating token embeddings for testing data
Testing_seq = token.texts_to_sequences(X_train)
Testing_pad = pad_sequences(Testing_seq, maxlen=50, padding='post', truncating='post')
Y_val_onehot = keras.utils.to_categorical(Y_val, num_classes=3)

# Pad sequences to a fixed length
max_sequence_length = 50  # Adjust based on your maximum sequence length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_length)
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)



# Model Architecture
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=100, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # Three units for three classes and softmax activation

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

model.build()
# Define EarlyStopping callback
es = EarlyStopping(patience=2,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)

lr = ReduceLROnPlateau(patience = 2,
                       monitor = 'val_loss',
                       factor = 0.5,
                       verbose = 0)




# Fit the model
history = model.fit(X_train_padded, Y_train,
                    validation_data=(X_val_padded, Y_val),
                    epochs=50,
                    batch_size=50,
                    callbacks=[es],
                    verbose=1)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
plt.show()

def preprocess_input(text):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence

# Function to predict the class
def predict_class(user_input):
    # Preprocess the input
    processed_input = preprocess_input(user_input)

    # Make a prediction using the model
    prediction = model.predict(processed_input)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Map predicted class to human-readable label
    class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    predicted_label = class_labels[predicted_class]

    return predicted_label

def predict_hate_speech(text):
    text = preprocess_input(text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(sequence)[0]
    predicted_class = np.argmax(prediction)
    return predicted_class

def on_send():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter some text")
        return

    # Get the current time for the timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Display user input in chat window
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You ({timestamp}): {user_input}\n")
    chat_window.config(state=tk.DISABLED)

    # Clear the input field
    entry.delete(0, tk.END)

    # Get the prediction
    prediction = predict_class(user_input)

    # Display the bot's response
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"Bot ({timestamp}): {prediction}\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

def on_clear():
    chat_window.config(state=tk.NORMAL)
    chat_window.delete('1.0', tk.END)
    chat_window.config(state=tk.DISABLED)

# Create the main window
root = tk.Tk()
root.title("Hate Speech Detection Chatbot")
root.geometry("400x500")

# Apply some styling
root.configure(bg='#f0f0f0')
font_style = ("Helvetica", 12)

# Create and place the chat window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=50, height=20, font=font_style, bg='#ffffff')
chat_window.pack(pady=10, padx=10)

# Create and place the input field
entry = tk.Entry(root, width=50, font=font_style, bg='#ffffff')
entry.pack(pady=5, padx=10)

# Create a frame for the buttons
button_frame = tk.Frame(root, bg='#f0f0f0')
button_frame.pack(pady=5)

# Create and place the Send button
send_button = tk.Button(button_frame, text="Send", command=on_send, font=font_style, bg='#4CAF50', fg='#ffffff')
send_button.pack(side=tk.LEFT, padx=10)

# Create and place the Clear button
clear_button = tk.Button(button_frame, text="Clear", command=on_clear, font=font_style, bg='#f44336', fg='#ffffff')
clear_button.pack(side=tk.RIGHT, padx=10)

# Run the application
root.mainloop()


with open('hate_speech_detection_model.pickle', 'wb') as f:
    pickle.dump(model, f)



