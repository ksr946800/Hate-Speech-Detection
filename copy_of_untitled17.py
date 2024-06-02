
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from gensim.models import KeyedVectors

# Load pre-trained word embeddings (e.g., Word2Vec)
word_vectors = KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)

# Sample dataset (replace with your own labeled dataset)
X = np.array(['Bevarsi', 
'Bewarsi',
'Gandu',
'Chutiya',
'Lowda',
'Mindri',
'Magane',
'Loosu',
'Shata',
'Tunne unnu',
'Tika keisko',
'Tika Densko',
'Tika',
'Dengbeda',
'Nin amman',
'Nin akkan',
'Keytini',
'Dengtini',
'Jhatt',
'Soole munde',
'Dagar',
'Choolu',
'Huchnanmagne',
'Soole magne',
'Nin ajji',
'Nin ajja',
'Bosadimude',
'Bolimaga',
'Suley'])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Labels: 1 for offensive, 0 for non-offensive

# Convert words to embeddings
X_emb = np.array([word_vectors[word] if word in word_vectors else np.zeros(word_vectors.vector_size) for word in X])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_emb, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Make predictions
new_sentences = ['This is a rude comment', 'She gave a nice compliment']
new_sentences_emb = np.array([word_vectors[word] if word in word_vectors else np.zeros(word_vectors.vector_size) for word in new_sentences])
predictions = model.predict(new_sentences_emb)
print("Predictions:", predictions)