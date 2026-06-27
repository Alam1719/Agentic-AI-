import streamlit as st

from src.preprocess import load_and_preprocess_data
from src.recommender import BookRecommender
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="centered"
)
st.title("📚 Book Recommendation System")
st.write(
    "Select a book and get recommendations based on similar books."
)
df = load_and_preprocess_data(
    "data/goodreads_books.csv"
)

recommender = BookRecommender(df)

selected_book = st.selectbox(
    "Choose a book:",
    df['title'].values
)

if st.button("Recommend"):
    recommendations = recommender.recommend(
        selected_book
    )
    st.subheader(
        "Recommended Books"
    )

    for i, book in enumerate(
        recommendations,
        start=1
    ):
        st.write(
            f"### {i}. {book['title']}"
        )
        st.write(
            f"**Author:** {book['author']}"
        )
        st.write(
            f"**Similarity Score:** {book['score']}"
        )
        st.divider()
    
