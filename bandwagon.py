# import libraries
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import DBSCAN

# ================== import dataset ==================
# import dataset: rating.csv and movies.csv
ratings = pd.read_csv(r'C:\Users\Radovanovic\OneDrive\Radna površina\petnica\ml-latest-small\ratings.csv')
movies = pd.read_csv(r'C:\Users\Radovanovic\OneDrive\Radna površina\petnica\ml-latest-small\movies.csv')

# join tables and remove unnecessary columns (for now)
dataset = pd.merge(ratings, movies, on="movieId", how="inner")

# display dataset
pd.set_option('display.max_columns', len(dataset.columns))
print("Dataset is: ")
print(dataset)

# ================== metrics ==================
# calculate average rating
average_rating = dataset["rating"].mean()
average_rating = format(average_rating, ".1f")
print("Average rating: ")
print(average_rating)

# calculate user engagement
calculate_ratings_per_user = ratings["userId"].value_counts()
print("Ratings per user: ")
print(calculate_ratings_per_user)

# show rating trends over time
# convert the 'timestamp' column to datetime
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], unit='s')

# group ratings by year and calculate the average rating per year
ratings_by_year = dataset.groupby(dataset['timestamp'].dt.year)['rating'].mean()

# plot the rating trends over time
plt.figure(figsize=(10, 6))
plt.plot(ratings_by_year.index, ratings_by_year.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.title('Rating Trends Over Time')
plt.grid(True)
plt.show()

# remove unnecessary columns (for now)
columns_to_remove = ['timestamp', 'title', 'genres']
dataset = dataset.drop(columns=columns_to_remove, axis=1)

pd.set_option('display.max_columns', len(dataset.columns))
print("Dataset is: ")
print(dataset)

# ================== import reverse bandwagon attack ==================
# number of random users
number_users = 1000
# rating per user
ratings_per_user = 20
# low rating
low_ratings = [1.0, 1.5, 2.0]
# item id of the target item
target_item = 39

# generate random user IDs
random_user_ids = np.arange(611, 1000 + number_users)

# create a new dataframe for the reversse bandwagon attack
attack_data = []

for user_id in random_user_ids:
    # generate random low ratings for items
    low_rated_items = np.random.choice(dataset['movieId'], size=5, replace=False)

    # set a low rating for the target item
    low_rated_items = np.append(low_rated_items, target_item)

    # create data for each low rated item
    for item_id in low_rated_items:
        # choose random value 1.0, 1.5 or 2.0
        low_rating = np.random.choice(low_ratings)
        attack_data.append({'userId': user_id, 'movieId': item_id, 'rating': low_rating})

# create a dataframe from the attack data
attack_dataset = pd.DataFrame(attack_data)

# combine the original dataset with the attack dataset
combined_dataset = pd.concat([dataset, attack_dataset], ignore_index=True)

# save the combined dataset to a new file
combined_dataset.to_csv('reverse_bandwagon_attack_dataset.csv', index=False)

# check if there was bandwagon attack
reverse_bandwagon_attack_dataset = pd.read_csv(r'C:\Users\Radovanovic\OneDrive\Radna površina\petnica\reverse_bandwagon_attack_dataset.csv')
check_average = reverse_bandwagon_attack_dataset["rating"].mean()
check_average = format(check_average, ".1f")
print("Check average: ")
print(check_average)

# ================== data cleaning ==================
# checking if there are missing values --> there are not
missing_values = dataset.isnull().sum()
print(f"Number of missing values: {missing_values}")

# checking if there are duplicate rows --> there are not
duplicates = dataset.duplicated().sum()
print(f"Duplicate rows in dataset: {duplicates}")

# ================== create user vectors based on their movie ratings ==================
# create a user-movie rating matrix
# create a mapping of userId and movieId to integer indices
user_ids = dataset['userId'].unique()
movie_ids = dataset['movieId'].unique()
user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

# create a sparse matrix to store user-movie ratings
user_movie_matrix = dok_matrix((len(user_ids), len(movie_ids)), dtype=np.float32)

# fill in the matrix with ratings
for _, row in dataset.iterrows():
    user_idx = user_id_to_idx[row['userId']]
    movie_idx = movie_id_to_idx[row['movieId']]
    user_movie_matrix[user_idx, movie_idx] = row['rating']

# convert the dok_matrix to a more memory-efficient format
user_movie_matrix = user_movie_matrix.tocsr()

# vectorize the user-movie rating matrix
user_vectors = user_movie_matrix.toarray()

# user_idx = user_id_to_idx[4]
# user_vector = user_vectors[user_idx]

"""
not random target and selected item
target_genre = 'Romance'
romance_movies = dataset[dataset['genres'].str.contains(target_genre, case=False)]
print(romance_movies)
"""