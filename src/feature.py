from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')

def make_features(df):
    vectorizer = CountVectorizer(
        lowercase=True, 
        stop_words=stopwords.words('french'),# choose the language that you hopefully understand ;) 
        max_df=0.95,  # Ignore words that appear in more than 95% of documents
        min_df=2  # Ignore words that appear in less than 2 documents
    )

    # Target variable
    y = df["is_comic"]

    # Applying transformations & encoding
    X = vectorizer.fit_transform(df["video_name"])

    return X, y
