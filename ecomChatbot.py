import random  # Importing the random module

import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Load intents from JSON (correcting the path)
intents = json.loads(open('intents.json').read())

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

random.shuffle(documents)

# Initialize lists for training
train_x = []
train_y = []

# Generate training data
for doc in documents:
    bag = [0] * len(words)
    pattern_words = doc[0]
    intent = doc[1]
    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1
    train_x.append(bag)
    label = [0] * len(classes)
    label[classes.index(intent)] = 1
    train_y.append(label)

# Convert lists to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

# Define and compile the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model to the data
history = model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.h5')

print("Model training completed. Model saved as 'chatbot_model.h5'.")

# Load the model and necessary files
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

