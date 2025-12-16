# Generated from: Week08_Final.ipynb
# Converted at: 2025-12-16T06:27:05.472Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import tensorflow as tf
import pandas as pd
import numpy as np

def read_gz_file(file_path, low_memory=True):
    dataset = pd.read_csv(file_path, sep='\t', low_memory=low_memory)
    return dataset
def save_dataframe_to_file(df, filename):
    df.to_csv(filename, index=False)
def save_ndarry_to_file(data, filename):
    np.savetxt(filename, data, delimiter=',')
def load_file_to_dataframe(filename):
    return pd.read_csv(filename)
def load_file_to_ndarry(filename):
    return pd.read_csv(filename).values

# ## Prepare data, process each file, clean up null values, cut out unnecessary features, and clean up data and features


# Create a data frame object to read the compressed file
title_basics_tsv_df = read_gz_file('title.basics.tsv',low_memory=False)
# Remove noise
title_basics_tsv_df = title_basics_tsv_df.drop(columns=['startYear', 'endYear', 'runtimeMinutes'])
# Remove rows where the title column has missing values
title_basics_tsv_df = title_basics_tsv_df.dropna(subset=['titleType','primaryTitle','originalTitle','isAdult','genres'])
# Filter for movies' tconst
title_basics_tsv_df = title_basics_tsv_df[title_basics_tsv_df['titleType'] == 'movie']
title_basics_tsv_df = title_basics_tsv_df.drop(columns=['titleType'])
# Display the first few rows of data
print(title_basics_tsv_df.head(10))
# Build a tconst set
movie_tconsts = set(title_basics_tsv_df['tconst'])
# Build a filter
def filter_by_tconst(df, tconst_column):
    return df[df[tconst_column].isin(movie_tconsts)]
# Build a filter
def clean_known_for_titles(df, movie_tconsts):
    def filter_known_for_titles(row):
        # Split the string into an array
        titles = row['knownForTitles'].split(',')
        # Filter out tconsts that are out of range
        filtered_titles = [title for title in titles if title in movie_tconsts]
        # Overwrite the new data back into knownForTitles
        row['knownForTitles'] = ','.join(filtered_titles)
        # Return this row
        return row

    # Filter and clean each row of data
    df = df.apply(filter_known_for_titles, axis=1)
    # Remove rows where knownForTitles is empty, which indicates that the row is completely out of range
    df = df[df['knownForTitles'] != '']
    return df

# Create a data frame object to read the compressed file
name_basics_tsv_df = read_gz_file('name.basics.tsv')
# Remove noise
name_basics_tsv_df = name_basics_tsv_df.drop(columns=['primaryName', 'birthYear', 'deathYear'])
name_basics_tsv_df = name_basics_tsv_df.dropna(subset=['nconst','primaryProfession','knownForTitles'])
# Filter and re-clean
name_basics_tsv_df = clean_known_for_titles(name_basics_tsv_df, movie_tconsts)
# Build an nconst set
name_nconsts = set(name_basics_tsv_df['nconst'])
# Display the first few rows of data
print(name_basics_tsv_df.head(10))
# Build a filter
def filter_by_nconst(df, nconst_column):
    return df[df[nconst_column].isin(name_nconsts)]

# Create a data frame object to read the compressed file
title_akas_tsv_df = read_gz_file('title.akas.tsv')
# Remove noise
title_akas_tsv_df = title_akas_tsv_df.drop(columns=['ordering', 'language', 'attributes'])
# Remove rows where the title column has missing values
title_akas_tsv_df = title_akas_tsv_df.dropna(subset=['title','titleId'])
# Re-clean the data
title_akas_tsv_df = filter_by_tconst(title_akas_tsv_df, 'titleId')
title_akas_tsv_df = title_akas_tsv_df.rename(columns={'titleId': 'tconst'})
# Display the first few rows of data
print(title_akas_tsv_df.head(10))

def filter_nconst_directors(row):
    if row['directors'] == '\\N':
        return row['directors']
    names = row['directors'].split(',')
    filtered_names = [name for name in names if name in name_nconsts]
    return ','.join(filtered_names) if filtered_names else '\\N'
def filter_nconst_writers(row):
    if row['writers'] == '\\N':
        return row['writers']
    names = row['writers'].split(',')
    filtered_names = [name for name in names if name in name_nconsts]
    return ','.join(filtered_names) if filtered_names else '\\N'
# Create a data frame object to read the compressed file
title_crew_tsv_df= read_gz_file('title.crew.tsv')
# Re-clean the data
title_crew_tsv_df = filter_by_tconst(title_crew_tsv_df, 'tconst')
title_crew_tsv_df['directors'] = title_crew_tsv_df.apply(filter_nconst_directors, axis=1)
title_crew_tsv_df['writers'] = title_crew_tsv_df.apply(filter_nconst_writers, axis=1)

title_crew_tsv_df = title_crew_tsv_df.rename(columns={'directors': 'directors_nconst'})
title_crew_tsv_df = title_crew_tsv_df.rename(columns={'writers': 'writers_nconst'})
# Display the first few rows of data
print(title_crew_tsv_df.head(10))

# Create a data frame object to read the compressed file
title_ratings_tsv_df= read_gz_file('title.ratings.tsv')
# Remove rows where the directors column has missing values
title_ratings_tsv_df = title_ratings_tsv_df.dropna(subset=['averageRating','numVotes'])
# Re-clean the data
title_ratings_tsv_df = filter_by_tconst(title_ratings_tsv_df, 'tconst')
# Display the first few rows of data
print(title_ratings_tsv_df.head(10))

## Save these DataFrame data to files for future reading
save_dataframe_to_file(title_basics_tsv_df,'title_basics_tsv_df.csv')
save_dataframe_to_file(name_basics_tsv_df,'name_basics_tsv_df.csv')
save_dataframe_to_file(title_akas_tsv_df,'title_akas_tsv_df.csv')
save_dataframe_to_file(title_crew_tsv_df,'title_crew_tsv_df.csv')
save_dataframe_to_file(title_ratings_tsv_df,'title_ratings_tsv_df.csv')
del title_basics_tsv_df
del name_basics_tsv_df
del title_akas_tsv_df
del title_crew_tsv_df
del title_ratings_tsv_df

# ## Link tables to create data tables with tighter information density


# ## The code here finally merges three new tables


# Read the data
title_basics_tsv_df = load_file_to_dataframe('title_basics_tsv_df.csv')
title_basics_tsv_df = title_basics_tsv_df.drop(columns=['primaryTitle', 'originalTitle'])
name_basics_tsv_df = load_file_to_dataframe('name_basics_tsv_df.csv')
title_akas_tsv_df = load_file_to_dataframe('title_akas_tsv_df.csv')
title_crew_tsv_df = load_file_to_dataframe('title_crew_tsv_df.csv')
title_ratings_tsv_df = load_file_to_dataframe('title_ratings_tsv_df.csv')

# Create a comprehensive movie information feature table
# Filter original titles
original_titles_df = title_akas_tsv_df[title_akas_tsv_df['isOriginalTitle'] == 1]

# Merge the title_basics_tsv_df table and the Original Titles table
movies_info_df = pd.merge(title_basics_tsv_df, original_titles_df[['tconst', 'title']], on='tconst', how='left')

# Merge the title_ratings_tsv_df table
movies_info_df = pd.merge(movies_info_df, title_ratings_tsv_df[['tconst', 'averageRating', 'numVotes']], on='tconst', how='left')

# Replace missing values in averageRating and numVotes with the median
average_rating_median = movies_info_df['averageRating'].median()
num_votes_median = movies_info_df['numVotes'].median()

movies_info_df['averageRating'].fillna(average_rating_median, inplace=True)
movies_info_df['numVotes'].fillna(num_votes_median, inplace=True)

# Round averageRating to an integer
movies_info_df['averageRating'] = movies_info_df['averageRating'].round().astype(int)

# Convert numVotes to the nearest multiple of 10
movies_info_df['numVotes'] = movies_info_df['numVotes'].apply(lambda x: int(np.round(x / 10) * 10))

# Select the required columns
movies_info_df[['genres1', 'genres2', 'genres3']] = movies_info_df['genres'].str.split(',', expand=True).fillna('\\N')
movies_info_df = movies_info_df[['tconst','title','genres1','genres2','genres3','isAdult','averageRating','numVotes']]

# Convert isAdult, averageRating, numVotes columns to string type
movies_info_df['averageRating'] = movies_info_df['averageRating'].astype(str)
movies_info_df['numVotes'] = movies_info_df['numVotes'].astype(str)

print(movies_info_df.head())

save_dataframe_to_file(movies_info_df,'movies_info_df.csv')

# Assuming you have loaded the data into the following DataFrame
name_basics_tsv_df = pd.read_csv('name_basics_tsv_df.csv')
title_crew_tsv_df = pd.read_csv('title_crew_tsv_df.csv')

# Split the primaryProfession column and the knownForTitles column by commas and expand them into multiple columns
name_basics_tsv_df[['primaryProfession_top1', 'primaryProfession_top2', 'primaryProfession_top3']] = name_basics_tsv_df['primaryProfession'].str.split(',', expand=True).fillna('\\N')
name_basics_tsv_df[['knownForTitle1', 'knownForTitle2', 'knownForTitle3', 'knownForTitle4']] = name_basics_tsv_df['knownForTitles'].str.split(',', expand=True).fillna('\\N')

# Initialize isDirectors and isWriters columns as 0
name_basics_tsv_df['isDirectors'] = 0
name_basics_tsv_df['isWriters'] = 0

# Set the isDirectors column
directors = title_crew_tsv_df['directors_nconst'].str.split(',').explode().unique()
name_basics_tsv_df.loc[name_basics_tsv_df['nconst'].isin(directors), 'isDirectors'] = 1

# Set the isWriters column
writers = title_crew_tsv_df['writers_nconst'].str.split(',').explode().unique()
name_basics_tsv_df.loc[name_basics_tsv_df['nconst'].isin(writers), 'isWriters'] = 1

# Keep the required columns
staff_df = name_basics_tsv_df[['nconst', 'primaryProfession_top1', 'primaryProfession_top2', 'primaryProfession_top3', 'knownForTitle1', 'knownForTitle2', 'knownForTitle3', 'knownForTitle4', 'isDirectors', 'isWriters']]

print(staff_df.head())

save_dataframe_to_file(staff_df,'staff_df.csv')

# Select the required columns
regional_titles_df = title_akas_tsv_df[['tconst', 'title', 'region', 'types']]
print(regional_titles_df.head())
save_dataframe_to_file(regional_titles_df,'regional_titles_df.csv')
del title_basics_tsv_df
del name_basics_tsv_df
del title_akas_tsv_df
del title_crew_tsv_df
del title_ratings_tsv_df

# ## Create a dictionary and perform one hot encoding (vectorization)


movies_info_df = load_file_to_dataframe('movies_info_df.csv')
regional_titles_df = load_file_to_dataframe('regional_titles_df.csv')
staff_df = load_file_to_dataframe('staff_df.csv')

import random

random.seed(42)
vocab = {}
global_counter = 0
vocab['0'] = 0
vocab['1'] = 1
global_counter = 1

int_list_record = set([0,1])
movies_info_df_averageRating_list = set(movies_info_df['averageRating'].tolist())
movies_info_df_numVotes_list = set(movies_info_df['numVotes'].tolist())
unique_int_set = int_list_record.union(movies_info_df_averageRating_list.union(movies_info_df_numVotes_list))

def get_valid_counter_num():
    global global_counter
    while global_counter in unique_int_set:
        global_counter += 1
    return global_counter

def register_vocab(data_list):
    global global_counter
    for item in data_list:
        if item not in vocab:
            vocab[item] = get_valid_counter_num()
            global_counter += 1

def record(df, col_name):
    col_list = df[col_name].tolist()
    random.shuffle(col_list)
    register_vocab(col_list)

# ### Record


# #### This recording process has a sequential requirement, so nconst and tconst are sequential in the original data.


# #### Therefore, in this step, I did not record nconst and tconst first, but first recorded some information that can represent classification information, such as region and type, etc.


# #### Finally, we record tconst and nconst, and some very isolated data such as title.


# #### In all the above recording processes, random.shuffle is used to ensure that the data does not have any sequential representation.


record(movies_info_df, 'genres1')
record(movies_info_df, 'genres2')
record(movies_info_df, 'genres3')
record(regional_titles_df, 'region')
record(regional_titles_df, 'types')
record(staff_df, 'primaryProfession_top1')
record(staff_df, 'primaryProfession_top2')
record(staff_df, 'primaryProfession_top3')
record(movies_info_df, 'tconst')
record(staff_df, 'nconst')
record(movies_info_df, 'title')
record(regional_titles_df, 'title')

# ### Save Dictionary


import csv
# Save
with open('vocab.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in vocab.items():
        writer.writerow([key, value])

# ### Linking the movies and regional tables


from concurrent.futures import ThreadPoolExecutor
import tempfile

# Assuming staff_df and regional_titles_df have been loaded and are available
# staff_df = pd.read_csv('path_to_staff_df.csv')
# regional_titles_df = pd.read_csv('path_to_regional_titles_df.csv')

def process_batch(temp_file_name, movies_info_batch, regional_titles_df):
    regional_titles_batch = regional_titles_df.iloc[temp_file_name]
    
    for index, row in regional_titles_batch.iterrows():
        tconst = row['tconst']
        region = row['region']
        movie_type = row['types']

        if tconst in movies_info_batch['tconst'].values:
            movies_info_batch.loc[movies_info_batch['tconst'] == tconst, f'region_class_{region}'] = 1
            movies_info_batch.loc[movies_info_batch['tconst'] == tconst, f'movie_type_{movie_type}'] = 1

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    movies_info_batch.to_pickle(temp_file.name)
    return temp_file.name

def combine_temp_files(temp_files):
    combined_df = pd.concat([pd.read_pickle(temp_file) for temp_file in temp_files], ignore_index=True)
    return combined_df

def batch_transform(data, batch_size=10000):
    temp_files = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, data.shape[0], batch_size):
            batch_indices = range(i, min(i + batch_size, data.shape[0]))
            futures.append(executor.submit(process_batch, batch_indices, data.iloc[batch_indices], regional_titles_df))

        for future in futures:
            temp_files.append(future.result())

    return temp_files

# Initialize new columns
regional_titles_df['region'] = regional_titles_df['region'].astype(str)
regional_titles_df['types'] = regional_titles_df['types'].astype(str)
unique_regions = sorted(regional_titles_df['region'].unique())
unique_types = sorted(regional_titles_df['types'].unique())

# Create a DataFrame with all the new columns
new_columns = {f'region_class_{region}': 0 for region in unique_regions}
new_columns.update({f'movie_type_{movie_type}': 0 for movie_type in unique_types})
new_columns_df = pd.DataFrame(new_columns, index=movies_info_df.index)

# Merge new columns into movies_info_df
movies_info_df = pd.concat([movies_info_df, new_columns_df], axis=1)

# Batch process and save to disk
temp_files = batch_transform(movies_info_df)

# Load from disk and merge results
final_df = combine_temp_files(temp_files)
 
print(final_df.head())
save_dataframe_to_file(final_df, 'movies_info_regional_combine_df.csv')

# ### Link movies_info_regional_combine and staff tables (get final table)


staff = load_file_to_dataframe('staff_df.csv')

# Expand the four knownForTitle columns
staff_melted = staff.melt(id_vars=['nconst'], 
                          value_vars=['knownForTitle1', 'knownForTitle2', 'knownForTitle3', 'knownForTitle4'],
                          var_name='knownForTitleNumber', 
                          value_name='knownForTitle')

# Drop rows with missing values
staff_melted = staff_melted.dropna(subset=['knownForTitle'])

# Select the required columns
staff_expanded = staff_melted[['nconst', 'knownForTitle']]

# Read the data
movies_info_regional_combine_df = load_file_to_dataframe('movies_info_regional_combine_df.csv')
merged_df = staff_expanded.merge(movies_info_regional_combine_df, left_on='knownForTitle', right_on='tconst', how='inner')

# Save the final table
merged_df = merged_df.drop(['knownForTitle'], axis=1)
print(merged_df.head())
save_dataframe_to_file(merged_df, 'final_merged_staff_movies_info.csv')

# ### Vectorized Coding


vocab_df = pd.read_csv('vocab.csv', header=None)  # Ensure no header row
vocab_df.columns = ['key', 'value']  # Add column names for easier handling
# Convert DataFrame to dictionary
vocab = pd.Series(vocab_df['value'].values, index=vocab_df['key']).to_dict()

merged_df = load_file_to_dataframe('final_merged_staff_movies_info.csv')

# Define a function to map the columns
def vectorize_column(column, vocab):
    return column.apply(lambda x: vocab.get(x, x))

# Iterate over all columns of merged_df
for col in merged_df.columns:
    if merged_df[col].dtype != 'int64':
        merged_df[col] = vectorize_column(merged_df[col], vocab)

# Print the result
print(merged_df.head())
# Save the vectorized table
save_dataframe_to_file(merged_df, 'final_mapped_vec.csv')

# ## Feature Fusion


# #### Because the internal features of the original data are unknown after vectorization, such as their clustering, their feature correlation, etc., through the unsupervised learning of the neural network, the features of all vectors are fused and the final fused features are obtained.


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Read the CSV file
final_mapped_vec_df = pd.read_csv('final_mapped_vec.csv')

# Ensure final_mapped_vec is a numpy array
final_mapped_vec = final_mapped_vec_df.values

# Check and adjust data shape
if final_mapped_vec.ndim == 1:
    final_mapped_vec = final_mapped_vec.reshape(-1, 1)

print(f"Array dimensions: {final_mapped_vec.ndim}")
print(f"Array shape: {final_mapped_vec.shape}")

# Print the first few rows of data to confirm
print(final_mapped_vec[:5])

# Standardize the data
scaler = StandardScaler()
final_mapped_vec = scaler.fit_transform(final_mapped_vec)

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# Build the model
input_layer = Input(shape=(final_mapped_vec.shape[1],))
dense_layer_1 = Dense(512, activation='relu')(input_layer)  # High-dimensional features
dropout_layer1 = Dropout(0.25)(dense_layer_1)
dense_layer_2 = Dense(256, activation='relu')(dropout_layer1)  # High-dimensional features
dropout_layer2 = Dropout(0.25)(dense_layer_2)
dense_layer_3 = Dense(128, activation='relu')(dropout_layer2)  # High-dimensional features
dropout_layer3 = Dropout(0.25)(dense_layer_3)
output_layer = Dense(final_mapped_vec.shape[1], activation='relu')(dropout_layer3)  # Fused features

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mse')
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
history = model.fit(final_mapped_vec,
                    final_mapped_vec,
                    epochs=100,
                    batch_size=2048,
                    validation_split=0.2,
                    callbacks=[early_stopping, checkpoint])

# Load the best model weights
model.load_weights('best_model.keras')

# Get the fused features
fused_features = model.predict(final_mapped_vec)

# Save the fused features to a Parquet file
fused_features_df = pd.DataFrame(fused_features)
fused_features_df.to_parquet('fused_features.parquet', index=False)
print("Fused features saved to fused_features.parquet")

# ### The fused features are used as input into Wrod2Vec to obtain the final word embedding


import os
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tqdm
import re
import string
import io
from sklearn.model_selection import train_test_split
import pandas as pd

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim)

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots

def export(word2vec, vocab):
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()

    try:
        from google.colab import files
        files.download('vectors.tsv')
        files.download('metadata.tsv')
    except Exception:
        pass
    return weights

def movie_recommendation_word2vec(fused_features, max_features=20000, window_size=2, num_ns=20, BATCH_SIZE=1024, BUFFER_SIZE=10000, embedding_dim=150, epochs=25, SEED=42):
    sequences = fused_features
    vocab_size = max_features

    # Generate training data
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=window_size,
        num_ns=num_ns,
        vocab_size=vocab_size,
        seed=SEED)
    
    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    AUTOTUNE = tf.data.AUTOTUNE

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback])

    # Export word vectors
    vocab = [str(i) for i in range(vocab_size)]  # Construct a simple vocabulary
    weights = export(word2vec, vocab)

    return weights

# Assuming fused_features are the fused features
fused_features = pd.read_parquet('fused_features.parquet').values
print("Read fused features done")
fused_features = tf.data.Dataset.from_tensor_slices(fused_features.astype(int))
fused_features = list(fused_features.as_numpy_iterator())

# Train Word2Vec model
weights = movie_recommendation_word2vec(fused_features)