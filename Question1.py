# Question1
# ---------

import json
import pickle
import numpy as np

# Cleaning the Dataset
# ---------------------

"""
with open('Datasets/Data.json', 'r') as file:
    dishes = json.load(file)

unique_dishes = []
seen_dishes = set()

for dish in dishes:
    dish_name = dish["Item"]
    # Check if the dish is not seen before
    if dish_name not in seen_dishes:
        unique_dishes.append(dish)
        seen_dishes.add(dish_name)

with open('Datasets/Data1.json', 'w') as file:
    json.dump(unique_dishes, file, indent=2)"""


# Counting the number of unique dishes
# -------------------------------------

"""
with open('Datasets/Data1.json', 'r') as file:
    dishes = json.load(file)
unique_dishes = len(dishes)
print(f"Number of unique dishes: {unique_dishes}")"""

# -------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
with open('Datasets/Data1.json', 'r') as file:
    dataset = json.load(file)

# Extract features and labels
X = [item["Item"] for item in dataset]
y = [item["Categories"] for item in dataset]

# Convert labels to a binary matrix
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Build a more complex neural network
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train_tfidf.shape[1]))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(y_binary.shape[1], activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train_tfidf, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
accuracy = model.evaluate(X_test_tfidf, y_test)[1]
print("Test Accuracy:", accuracy)

# Save the trained model to a file in Keras format
model.save('neural_network_model.keras')