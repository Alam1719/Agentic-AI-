import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:

    def __init__(self, dataframe):

        self.df = dataframe

        self.vectorizer = TfidfVectorizer(
            stop_words='english'
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.df['tags']
            )
        )

        self.similarity_matrix = (
            cosine_similarity(
                self.tfidf_matrix
            )
        )

        self.indices = pd.Series(
            self.df.index,
            index=self.df['title']
        ).drop_duplicates()

    def recommend(self, book_title, top_n=5):

        if book_title not in self.indices:
            return []

        idx = self.indices[book_title]

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[idx]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = (
            similarity_scores[1:top_n + 1]
        )

        recommendations = []

        for i, score in similarity_scores:

            recommendations.append(
                {
                    "title":
                        self.df.iloc[i]['title'],
                    "author":
                        self.df.iloc[i]['authors'],
                    "score":
                        round(score, 3)
                }
            )

        return recommendations