from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Download stopwords if not already available
nltk.download('stopwords')

def make_features(df):
    # Initialize the French Snowball Stemmer and NLTK French stop words list
    stemmer = SnowballStemmer("french")
    french_stop_words = stopwords.words("french")
    
    # Preprocess text to apply stemming
    df['video_name'] = df['video_name'].apply(
        lambda text: ' '.join(stemmer.stem(word) for word in text.split())
    )
    
    # Initialize CountVectorizer with NLTK stop words and document frequency limits
    vectorizer = CountVectorizer(
        lowercase=True, 
        stop_words=french_stop_words,  # Using comprehensive French stop words
        max_df=0.95,  # Tuning to ignore very common terms
        min_df=2      # Tuning to ignore rare terms
    )
    
    # Transform the video names and get the labels
    X = vectorizer.fit_transform(df["video_name"])
    y = df["is_comic"]

    return X, y, vectorizer


