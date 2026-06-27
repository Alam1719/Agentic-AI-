import pandas as pd


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the Goodreads dataset.
    """

    df = pd.read_csv(filepath, on_bad_lines='skip')

    df.columns = df.columns.str.strip()

    df = df.fillna('')

    df = df.drop_duplicates(subset='title')

    df['tags'] = (
        df['title'].astype(str) + ' ' +
        df['authors'].astype(str) + ' ' +
        df['publisher'].astype(str) + ' ' +
        df['language_code'].astype(str)
    )

    df = df[
        [
            'bookID',
            'title',
            'authors',
            'average_rating',
            'ratings_count',
            'tags'
        ]
    ]

    return df
  