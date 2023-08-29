# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# divide dataset in training set and test set
# split the main sataset into training and test sets (80%-20%)
train_data, test_data = train_test_split(reverse_bandwagon_attack_dataset, test_size=0.2, random_state=42)

# identify the last 8000 lines (generated users)
generated_users= reverse_bandwagon_attack_dataset.tail(8000)

# split the generated users into training and test sets (80%-20%)
train_generated, test_generated = train_test_split(generated_users, test_size=0.2, random_state=42)

# merge the train data with train generated, and test data with test generated
merged_train_data = pd.concat([train_data, train_generated])
merged_test_data = pd.concat([test_data, test_generated])

print("Train Data Size:", train_data.shape)
print("Test Data Size:", test_data.shape)
print("Train Generated Users Size:", train_generated.shape)
print("Test Generated Users Size:", test_generated.shape)
print("merged train data", merged_train_data.shape)
print("merged test data", merged_test_data.shape)

# ================== create user vectors based on their movie ratings ==================
# Create user-movie rating matrix
user_movie_matrix = pd.pivot_table(merged_train_data, values='rating', index='userId', columns='movieId', fill_value=0)
print("matrix")
print(user_movie_matrix)

# extract user rating vectors -> each vector represent user with all their ratings
user_rating_vectors = user_movie_matrix.values
print("vectors")
print(user_rating_vectors)

# normalize the user rating vectors
scaler = StandardScaler()
user_rating_vectors_normalized = scaler.fit_transform(user_rating_vectors)
print(user_rating_vectors_normalized)

# apply DBSCAN algorithm
eps = 0.1
min_samples = 3

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(user_rating_vectors_normalized)

# the labels array contains the cluster assignments for each user
# users with the same cluster label are part of the same cluster
# and users with -1 label are considered as noise

# vizualize the clusters
# data will be placed in 2D place
pca = PCA(n_components=2)
user_rating_vectors_reduced = pca.fit_transform(user_rating_vectors_normalized)

plt.scatter(user_rating_vectors_reduced[:, 0], user_rating_vectors_reduced[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering of User Rating Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



"""
not random target and selected item
target_genre = 'Romance'
romance_movies = dataset[dataset['genres'].str.contains(target_genre, case=False)]
print(romance_movies)
"""