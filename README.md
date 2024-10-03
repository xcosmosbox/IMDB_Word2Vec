# IMDB Movie Recommendation System By Word2Vec

This project is a movie recommendation system based on IMDB data, developed using Word2Vec and deep learning techniques. The goal is to embed movie features into a vector space and use these embeddings to recommend similar movies. The project also includes a visualization of movie embeddings using techniques such as PCA and UMAP.

## Project Structure

The workflow is as follows:

1. **Data Cleaning and Preprocessing**: Each table in the dataset undergoes extensive cleaning. Unnecessary columns are removed, missing values are handled, and key features are extracted.
2. **Feature Engineering**: Features such as movie genres, regions, types, and more are transformed into one-hot encoded vectors using a manually created vocabulary.
3. **Combining Data**: All the tables are merged into a final dataset, where each row represents a movie and its associated features.
4. **Embedding Features**: The final feature table is passed through a neural network, which is trained to learn the relationships between features. This process produces fused feature vectors that capture high-level information about each movie.
5. **Word2Vec Training**: Using the `Word2Vec` model, movie embeddings are further refined. By using a skip-gram approach with negative sampling, the model learns to place similar movies close together in the embedding space.
6. **Visualization**: The trained embeddings are visualized using PCA and UMAP, allowing us to see the relationships between movies in the latent space.

## PCA and UMAP Visualization

Below are the visualizations of the movie embeddings using PCA and UMAP:

### PCA Visualization

Due to limitations in Embedding Projector, only 2.4% of the data points are displayed.

![img_v3_02d7_b58791de-671a-47dd-8fa7-8c7a64061chu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/TiIQyMimg_v3_02d7_b58791de-671a-47dd-8fa7-8c7a64061chu.jpg)

### UMAP Visualization

This UMAP result shows 5000 data points, which is approximately 0.2% of the total sample size.

![img_v3_02d7_055b15a9-59b3-4d91-9ddd-08b74030a9hu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/yS2uKvimg_v3_02d7_055b15a9-59b3-4d91-9ddd-08b74030a9hu.jpg)

### Movie Embedding Exploration

The interactive visualization allows you to click on any point (a specific movie) and explore its neighbors—movies that are similar and recommended based on the embedding.

### Example of Movie Embedding Exploration

In this example, clicking on a movie reveals its neighboring points, representing other movies that are similar and recommended.

![img_v3_02d7_f6f00a71-a687-42f7-8f92-730e03e382hu](https://raw.githubusercontent.com/xavierfrankland/PicRepo/master/uPic/8kWCI2img_v3_02d7_f6f00a71-a687-42f7-8f92-730e03e382hu.jpg)

## Final Dataset

The final dataset consists of approximately 200,000 rows and 28 columns. The key feature table is created by one-hot encoding key attributes, like genres, movie types, and regions, using a manually constructed vocabulary.

### Final Table Size and Structure

The final dataset has undergone a detailed feature engineering process, with many unnecessary or redundant features removed. It contains crucial features like genres and regions, along with their one-hot encoded vectors, making it ideal for feeding into the neural network.

## Word2Vec Model

Once the final table was prepared, it was fed into a `Word2Vec` model. The purpose of this model was to further refine the relationships between movies by learning feature similarities and producing embeddings that could be used for movie recommendations.

### Word2Vec Results

The Word2Vec model provided high-dimensional movie embeddings. When examined through the Embedding Projector, the recommendations produced are highly accurate, with similar movies being placed close to each other in the vector space.