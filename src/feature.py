from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# Download stopwords if not already available
nltk.download('stopwords')

def make_features(df, use_tfidf=False):
    # Initialize the French Snowball Stemmer and NLTK French stop words list
    stemmer = SnowballStemmer("french")
    french_stop_words = stopwords.words("french")
    
    # Preprocess text to apply stemming
    df['video_name'] = df['video_name'].apply(
        lambda text: ' '.join(stemmer.stem(word) for word in text.split())
    )
    
    # Choose the vectorizer based on the use_tfidf flag
    vectorizer_class = TfidfVectorizer if use_tfidf else CountVectorizer
    vectorizer = vectorizer_class(
        lowercase=True, 
        stop_words=french_stop_words,
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2)  # Adding bigrams to capture more context
    )
    
    # Transform the video names and get the labels
    X = vectorizer.fit_transform(df["video_name"])
    y = df["is_comic"]

    return X, y, vectorizer
