# Question1.py
# ------------


# Cleaning the Dataset
# ---------------------

import json

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
    json.dump(unique_dishes, file, indent=2)

# ----------------------------------------------------------------
