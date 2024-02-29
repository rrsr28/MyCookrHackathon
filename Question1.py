# Question1
# ---------

import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input, concatenate

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Reading the models
# ------------------

"""with open('Models/Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('Models/ModelC.pkl', 'rb') as model_file:
    modelC = pickle.load(model_file)
with open('Models/ModelCu.pkl', 'rb') as model_file:
    modelCu = pickle.load(model_file)"""

with open('Models/mlb.pkl', 'rb') as model_file:
    mlb = pickle.load(model_file)
with open('Models/mlb_cuisine.pkl', 'rb') as model_file:
    mlb_cuisine = pickle.load(model_file)
with open('Models/mlb_course.pkl', 'rb') as model_file:
    mlb_course = pickle.load(model_file)


model = tf.keras.models.load_model('Models/Model.h5')
modelC = tf.keras.models.load_model('Models/ModelC.h5')
modelCu = tf.keras.models.load_model('Models/ModelCu.h5')

# --------------------------------------------------
    
# Reading the dataset
# -------------------
    
df = pd.read_csv('Datasets/IndianFoodDataset.csv')

# --------------------------------------------------


# Prediction
# ----------

max_words = 1500
tokenizer_ingredients = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer_ingredients.fit_on_texts(df['Ingredients'].astype(str))
tokenizer_recipe_name = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer_recipe_name.fit_on_texts(df['Name'].astype(str))
tokenizer_instructions = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer_instructions.fit_on_texts(df['Instructions'].astype(str))

# For already present dishes
new_input_ingredients = "1 1/2 cups Whole Wheat Flour,3/4 cup Jaggery - powdered,2 1/2 cups Water,1/2 teaspoon Cardamom Powder (Elaichi),Ghee - for cooking the dosa', '1 कप सफेद उरद दाल\xa0(split),1 कप पिली मूंग दाल,1 कप अरहर दाल,2 कप चावल,1/4 कप मेथी के दाने,1 प्याज,3 हरी मिर्च,1 टहनी कढ़ी पत्ता,1/4 कप नारियल,2 छोटे चमच्च जीरा,नमक - स्वाद अनुसार,5 टहनी हरा धनिया"
new_input_recipe_name = "vellai / jaggerya dosa recipe"
new_data_instructions = "To begin making the Vellai Dosa, in a large bowl, dissolve the powdered jaggery along with 1/2 cup water and allow it to rest for 30 minutes. After 30 minutes, strain the jaggery water to remove impurities.Add the wheat flour to the strained jaggery mixture and stir continuously until there are no lumps. Add remaining 2 cups water and mix to make a smooth batter. Toss in the cardamom powder and mix well. The next step is to cook the Vella Dosa on a skillet. Heat a skillet on medium heat and pour a ladleful of batter on the skillet. Swirl the pan in a circular motion to spread the batter around to make a thin crepe like dosa.Drizzle 1/4 teaspoon of ghee around the vella dosa and cook on low heat until well done. Flip over to cook the other side for a few seconds and serve warm.Serve Vellai Dosa as a delicious dish during Navratri or as breakfast along with Peanut Chutney."

answer = []

new_sequences_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
new_padded_ingredients = pad_sequences(new_sequences_ingredients, maxlen=150, padding='post', truncating='post')
new_sequences_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
new_padded_recipe_name = pad_sequences(new_sequences_recipe_name, maxlen=10, padding='post', truncating='post')
prediction = model.predict([new_padded_ingredients, new_padded_recipe_name])
threshold = 0.5
binary_prediction = (prediction > threshold).astype(int)
predicted_labels = mlb.inverse_transform(binary_prediction)
if(len(predicted_labels[0])):
  answer.append(predicted_labels[0][0])

sequences_new_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
padded_sequences_new_ingredients = pad_sequences(sequences_new_ingredients, maxlen=150, padding='post', truncating='post')
sequences_new_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
padded_sequences_new_recipe_name = pad_sequences(sequences_new_recipe_name, maxlen=10, padding='post', truncating='post')
sequences_new_instructions = tokenizer_instructions.texts_to_sequences([new_data_instructions])
padded_sequences_new_instructions = pad_sequences(sequences_new_instructions, maxlen=200, padding='post', truncating='post')
predictions = modelC.predict([padded_sequences_new_ingredients, padded_sequences_new_recipe_name, padded_sequences_new_instructions])
binary_predictions = (predictions > 0.5).astype(int)
predicted_classes = mlb_course.inverse_transform(binary_predictions)
if(len(predicted_classes[0])):
  answer.append(predicted_classes[0][0])

sequences_new_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
padded_sequences_new_ingredients = pad_sequences(sequences_new_ingredients, maxlen=150, padding='post', truncating='post')
sequences_new_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
padded_sequences_new_recipe_name = pad_sequences(sequences_new_recipe_name, maxlen=10, padding='post', truncating='post')
predictions = modelCu.predict([padded_sequences_new_ingredients, padded_sequences_new_recipe_name])
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
mlb_cuisine_original = mlb_cuisine
mlb_cuisine_eval = MultiLabelBinarizer(classes=mlb_cuisine_original.classes_)
mlb_cuisine_eval.fit_transform([[]])
predicted_classes = mlb_cuisine_eval.inverse_transform(binary_predictions)
if(len(predicted_classes[0])):
  answer.append(predicted_classes[0][0])

print(answer)

# -----------------

"""

dish_names = df["Name"].str.lower().tolist()

new_dish_name = input("Enter the name of the dish you want to find a similar recipe for: ").lower()

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(dish_names)
new_dish_vector = vectorizer.transform([new_dish_name])
similarities = cosine_similarity(new_dish_vector, tfidf_matrix)
top_n_indices = similarities.argsort()[0, -1:(-1 - 5):-1]

topDishes = []
most_similar_dishes = [dish_names[i] for i in top_n_indices]
print(f"The {5} most similar dishes in the dataset are:")
for dish in most_similar_dishes:
    topDishes.append(dish)
print("\n\nTop Dishes :\n")

recipes = []
inss = []
if(len(topDishes)):
  for DishDa in topDishes:
    recipe = df.loc[df["Name"].str.lower() == DishDa, "Ingredients"].values[0]
    ins = df.loc[df["Name"].str.lower() == DishDa, "Instructions"].values[0]
    recipes.append(recipe)
    inss.append(ins)

"""


"""

answer = []
if(len(topDishes) >= 5 and len(recipes) >= 5):

  for i in range(0, 5):

    new_input_ingredients = topDishes[i]
    new_input_recipe_name = new_dish_name
    new_data_instructions = inss[i]

    new_sequences_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
    new_padded_ingredients = pad_sequences(new_sequences_ingredients, maxlen=150, padding='post', truncating='post')
    new_sequences_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
    new_padded_recipe_name = pad_sequences(new_sequences_recipe_name, maxlen=10, padding='post', truncating='post')
    prediction = model.predict([new_padded_ingredients, new_padded_recipe_name])
    threshold = 0.5
    binary_prediction = (prediction > threshold).astype(int)
    predicted_labels = mlb.inverse_transform(binary_prediction)
    if(len(predicted_labels[0])):
      answer.append(predicted_labels[0][0])

    sequences_new_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
    padded_sequences_new_ingredients = pad_sequences(sequences_new_ingredients, maxlen=150, padding='post', truncating='post')
    sequences_new_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
    padded_sequences_new_recipe_name = pad_sequences(sequences_new_recipe_name, maxlen=10, padding='post', truncating='post')
    sequences_new_instructions = tokenizer_instructions.texts_to_sequences([new_data_instructions])
    padded_sequences_new_instructions = pad_sequences(sequences_new_instructions, maxlen=200, padding='post', truncating='post')
    predictions = modelC.predict([padded_sequences_new_ingredients, padded_sequences_new_recipe_name, padded_sequences_new_instructions])
    binary_predictions = (predictions > 0.5).astype(int)
    predicted_classes = mlb_course.inverse_transform(binary_predictions)
    if(len(predicted_classes[0])):
      answer.append(predicted_classes[0][0])

    sequences_new_ingredients = tokenizer_ingredients.texts_to_sequences([new_input_ingredients])
    padded_sequences_new_ingredients = pad_sequences(sequences_new_ingredients, maxlen=150, padding='post', truncating='post')
    sequences_new_recipe_name = tokenizer_recipe_name.texts_to_sequences([new_input_recipe_name])
    padded_sequences_new_recipe_name = pad_sequences(sequences_new_recipe_name, maxlen=10, padding='post', truncating='post')
    predictions = modelCu.predict([padded_sequences_new_ingredients, padded_sequences_new_recipe_name])
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    mlb_cuisine_original = mlb_cuisine
    mlb_cuisine_eval = MultiLabelBinarizer(classes=mlb_cuisine_original.classes_)
    mlb_cuisine_eval.fit_transform([[]])
    predicted_classes = mlb_cuisine_eval.inverse_transform(binary_predictions)
    if(len(predicted_classes[0])):
      answer.append(predicted_classes[0][0])

answer = list(set(answer))
print(answer)

"""