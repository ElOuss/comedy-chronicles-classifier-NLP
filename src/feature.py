from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def make_features(df):
    vectorizer = CountVectorizer(
        lowercase=True, 
        stop_words=stopwords.words('french'),
        max_df=0.95,
        min_df=2
    )

    # Transform the video names and get the labels
    X = vectorizer.fit_transform(df["video_name"])
    y = df["is_comic"]

    return X, y, vectorizer

