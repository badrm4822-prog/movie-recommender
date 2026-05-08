#!/usr/bin/env python
# coding: utf-8

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import streamlit as st


# Step 2: Read the datasets
BASE_DIR   = os.getcwd()
movies_df  = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
ratings_df = pd.read_csv(os.path.join(BASE_DIR, "ratings.csv"))

movies_missing  = movies_df.isnull().sum()
ratings_missing = ratings_df.isnull().sum()


# Step 3: Split the 'genres' column into separate columns and convert to boolean (True/False)
genres_split  = movies_df['genres'].str.split('|', expand=True)
unique_genres = set(genres_split.stack())

for genre in unique_genres:
    movies_df[genre] = movies_df['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)


# Step 4: Merge movies and ratings datasets on 'movieId'
merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')


# Step 5: Collaborative Filtering using SVD
reader   = Reader(rating_scale=(1, 5))
data     = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)
rmse_val    = accuracy.rmse(predictions)


# Step 6: Content-Based Filtering using TF-IDF and Cosine Similarity
movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x.split('|')))

tfidf        = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres_str'])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_movie_recommendations(movie_title, cosine_sim=cosine_sim):
    idx        = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]


# Step 7: Combine Collaborative Filtering and Content-Based Filtering
def get_collaborative_recommendations(user_id, movie_id, svd=svd, ratings_df=ratings_df):
    predicted_rating = svd.predict(user_id, movie_id).est
    return predicted_rating

def get_content_based_recommendations(movie_title, cosine_sim=cosine_sim):
    return get_movie_recommendations(movie_title)

def get_hybrid_recommendations(user_id, movie_title, alpha=0.5):
    content_based_recs = get_content_based_recommendations(movie_title)

    hybrid_list = []
    for title in content_based_recs:
        movie_id_row = movies_df[movies_df['title'] == title]
        if movie_id_row.empty:
            continue
        movie_id = movie_id_row.iloc[0]['movieId']

        cf_score      = get_collaborative_recommendations(user_id, movie_id, svd)
        content_score = cosine_sim[
            movies_df.index[movies_df['title'] == movie_title].tolist()[0],
            movies_df.index[movies_df['title'] == title].tolist()[0]
        ]

        hybrid_score = alpha * cf_score + (1 - alpha) * content_score * 5

        hybrid_list.append({
            "movie_title":   title,
            "cf_score":      round(cf_score, 4),
            "content_score": round(content_score, 4),
            "hybrid_score":  round(hybrid_score, 4)
        })

    hybrid_list = sorted(hybrid_list, key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_list


# Step 8: Evaluation
def evaluate_model(testset, svd=svd):
    actual_ratings    = [true_rating for (_, _, true_rating) in testset]
    predicted_ratings = [svd.predict(user_id, movie_id).est for (user_id, movie_id, _) in testset]

    rmse = np.sqrt(np.mean((np.array(actual_ratings) - np.array(predicted_ratings)) ** 2))
    mae  = np.mean(np.abs(np.array(actual_ratings) - np.array(predicted_ratings)))

    actual_binary    = [1 if rating >= 3 else 0 for rating in actual_ratings]
    predicted_binary = [1 if rating >= 3 else 0 for rating in predicted_ratings]

    precision = precision_score(actual_binary, predicted_binary)
    recall    = recall_score(actual_binary, predicted_binary)
    f1        = f1_score(actual_binary, predicted_binary)

    return rmse, mae, precision, recall, f1

rmse, mae, precision, recall, f1 = evaluate_model(testset)


# Step 9: Streamlit Interface
def streamlit_interface():
    st.title("Movie Recommendation System")

    # Rating Distribution Chart
    st.subheader("Rating Distribution")
    rating_counts = ratings_df['rating'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rating_counts.index, rating_counts.values, color='steelblue', width=0.4)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Ratings")
    ax.set_title("Distribution of Ratings in Dataset")
    ax.set_xticks(rating_counts.index)
    for i, v in enumerate(rating_counts.values):
        ax.text(rating_counts.index[i], v + 50, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)

    # Evaluation Metrics Chart
    st.subheader("Evaluation Metrics")
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    reg_labels = ['RMSE', 'MAE']
    reg_values = [rmse, mae]
    axes[0].bar(reg_labels, reg_values, color=['steelblue', 'coral'])
    axes[0].set_title('Regression Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, max(reg_values) * 1.4)
    for i, v in enumerate(reg_values):
        axes[0].text(i, v + 0.01, str(round(v, 4)), ha='center', fontweight='bold')

    clf_labels = ['Precision', 'Recall', 'F1-Score']
    clf_values = [precision, recall, f1]
    axes[1].bar(clf_labels, clf_values, color=['mediumseagreen', 'orchid', 'gold'])
    axes[1].set_title('Classification Metrics (threshold >= 3)')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.2)
    for i, v in enumerate(clf_values):
        axes[1].text(i, v + 0.01, str(round(v, 4)), ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig2)

    # Recommendations
    user_id     = st.number_input("Enter User ID", min_value=1, max_value=1000, value=1)
    movie_title = st.selectbox("Select Movie Title", movies_df['title'].values)

    if st.button('Get Recommendations'):
        hybrid_recs = get_hybrid_recommendations(user_id, movie_title)

        st.subheader("Collaborative Score for Selected Movie:")
        selected_movie_id = movies_df[movies_df['title'] == movie_title].iloc[0]['movieId']
        st.write(round(svd.predict(user_id, selected_movie_id).est, 4))

        st.subheader("Hybrid Recommendations:")
        recs_df = pd.DataFrame(hybrid_recs)
        st.dataframe(recs_df, use_container_width=True)

        st.subheader("Evaluation Metrics:")
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'Precision', 'Recall', 'F1-Score'],
            'Value':  [round(rmse, 4), round(mae, 4),
                       round(precision, 4), round(recall, 4), round(f1, 4)]
        })
        st.dataframe(metrics_df, use_container_width=True)

if __name__ == "__main__":
    streamlit_interface()
